#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO Post-Training for Qwen/Qwen2.5-0.5B-Instruct on MATH data.

- Uses TRL 0.21.0, Transformers 4.55.4, PEFT >= 0.10
- V100 friendly (fp16), LoRA optional.
- Dataset format: JSON with keys: idx, problem, solution, answer
"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import os, json, re
from typing import List, Dict, Any
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from trl import GRPOConfig, GRPOTrainer



# --------------------------
# Config (override via env)
# --------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_FILE  = os.environ.get("TRAIN_FILE", "/home/mohammad-m/TTT/Post_Training_Hybrid_RLSFT/MATH/data/train_data.json")
OUT_DIR    = os.environ.get("OUTPUT_DIR", "../saved_model/MATH/grpo_lora")
MERGED_DIR = os.environ.get("MERGED_DIR", "../saved_model/MATH/grpo_merged")

LR               = float(os.environ.get("LR", "5e-6"))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "16"))
ACCUM            = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
GROUP_SIZE       = int(os.environ.get("GROUP_SIZE", "4"))  # goes to Trainer (not Config) on 0.21.x


MAX_PROMPT_TOKENS     = int(os.environ.get("MAX_PROMPT_TOKENS", "768"))
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "512"))
DO_SAMPLE             = bool(os.environ.get("DO_SAMPLE", True))     # GRPO benefits from sampling
TEMPERATURE           = int(os.environ.get("TEMPERATURE", 1))
TOP_P                 = float(os.environ.get("TOP_P", "0.95"))
LOGGING_STEPS         = int(os.environ.get("LOGGING_STEPS", "5"))
SAVE_STEPS            = int(os.environ.get("SAVE_STEPS", "10000"))

# Optional LoRA: set USE_LORA=1 to turn on and adjust target modules if desired
USE_LORA = os.environ.get("USE_LORA", "1") == "1"
LORA_R   = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# --------------------------
# Load model / tokenizer
# --------------------------
def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    # Accelerate sets WORLD_SIZE; fall back to 1
    return int(os.environ.get("WORLD_SIZE", "1"))


tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype=torch.float16,
    device_map="auto",
)

model.generation_config = GenerationConfig(
    do_sample=DO_SAMPLE,     # GRPO benefits from sampling
    temperature=TEMPERATURE,
    top_p=TOP_P,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)

# --------------------------
# Data loading/helpers
# --------------------------
def _read_json_any(path: str):
    with open(path, "r") as f:
        text = f.read().strip()
        try:
            # JSON array
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception:
            # JSONL
            return [json.loads(line) for line in text.splitlines() if line.strip()]

def build_prompt(problem: str) -> str:
    # Instruction we want the model to always follow:
    user_msg = (
        "Solve the problem step by step. "
        "Show your reasoning, and finish with the final answer in \\boxed{...}.\n\n"
        f"Problem: {problem}"
    )
    messages = [
        {"role": "system", "content": "You are a meticulous math tutor. Always show your work."},
        {"role": "user", "content": user_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

raw = _read_json_any(DATA_FILE)

# map → {prompt, answer}
rows = []
for ex in raw:
    problem = str(ex["problem"]).strip()
    answer  = str(ex.get("answer", "")).strip()
    rows.append({
        "prompt": build_prompt(problem),
        "answer": answer,
    })

ds_all = Dataset.from_list(rows)
split = ds_all.train_test_split(test_size=0.05, seed=42)

# ---------------------------------------------------------------------------------------------------------
# Reward function (0.21.x)
# ---------------------------------------------------------------------------------------------------------
# In TRL 0.21.x, GRPO calls: reward_func(samples: List[str], **kwargs)
# It also forwards the original batch as kwargs["batch"] (dict of columns).
# --------------------------
# Reward function (0.21.x) — robust, step-aware, anti-repetition
# --------------------------
import math
import itertools

# 1) Regexes & tokenization helpers
boxed_re   = re.compile(r"\\boxed\{([^{}]+)\}")
brace_junk = re.compile(r"}\s*{")           # helps catch weird brace joins
rep4       = re.compile(r"(.)\1\1\1")       # any char repeated 4+
ops_re     = re.compile(r"(=|\\times|\\cdot|\\div|[+\-*/])")
kw_re      = re.compile(r"\b(thus|therefore|hence|so we|we have|it follows|then)\b", re.I)


def extract_boxed(text: str) -> str:
    """Return first \\boxed{...} content (raw, not normalized) or ''."""
    m = boxed_re.search(text)
    return m.group(1) if m else ""

def count_equations(s: str) -> int:
    return len(re.findall(r"=", s))

def count_ops(s: str) -> int:
    return len(ops_re.findall(s))

def count_reasoning_keywords(s: str) -> int:
    return len(kw_re.findall(s))

def word_tokens(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s)

def repetition_penalty(text: str) -> float:
    """Penalize obvious loops: long char runs, repeated trigrams, duplicated lines."""
    pen = 0.0
    # char runs like '.....' or 'aaaa'
    if rep4.search(text):
        pen -= 0.6
    # trigram repetition ratio
    toks = word_tokens(text.lower())
    if len(toks) >= 12:
        n = 3
        trigrams = list(zip(*(toks[i:] for i in range(n))))
        total = len(trigrams)
        uniq  = len(set(trigrams))
        dup_ratio = 0.0 if total == 0 else 1.0 - (uniq / total)
        # stronger penalty if lots of duplicated trigrams
        pen -= min(1.2, 3.0 * dup_ratio)
    # line duplication
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    if lines:
        freq = {}
        for ln in lines:
            freq[ln] = freq.get(ln, 0) + 1
        max_dup = max(freq.values())
        if max_dup >= 3:
            pen -= 0.6
        elif max_dup == 2 and len(lines) >= 6:
            pen -= 0.3
    return pen

def reasoning_score(pre_box: str) -> float:
    """Score genuine steps before the box: equations/ops/keywords + enough tokens."""
    toks = word_tokens(pre_box)
    eqs  = count_equations(pre_box)
    ops  = count_ops(pre_box)
    kws  = count_reasoning_keywords(pre_box)
    # Base signal from math-y structure
    structure = 0.6*eqs + 0.4*(ops//2) + 0.5*kws
    # Length gates: encourage non-trivial steps before the box
    len_bonus = 0.0
    if len(toks) >= 24:  len_bonus += 0.3
    if len(toks) >= 48:  len_bonus += 0.4
    if len(toks) >= 96:  len_bonus += 0.3
    # Cap total reasoning to keep magnitudes stable
    return min(2.2, structure + len_bonus)

def brevity_or_bloat_penalty(text: str, pre_box: str, first_box_pos: int) -> float:
    """Discourage 'just the answer' and extreme verbosity; also penalize 'answer-first'."""
    pen = 0.0
    words = len(word_tokens(text))
    pre_words = len(word_tokens(pre_box))
    # Too short overall → likely just the answer
    if words < 12:
        pen -= 0.6
    # If box appears ultra-early, it's answer-first
    if 0 <= first_box_pos <= 10:
        pen -= 0.5
    # Excessive verbosity (degenerate rambles)
    if words > 480:
        pen -= 0.6
    elif words > 320:
        pen -= 0.3
    # Too few words before the box (no real steps)
    if pre_words < 16:
        pen -= 0.3
    return pen

def _get_gold_answers_from_kwargs(**kwargs) -> List[str]:
    # TRL 0.21.x passes original batch under kwargs["batch"]; also support direct 'answer'
    if "batch" in kwargs and isinstance(kwargs["batch"], dict) and "answer" in kwargs["batch"]:
        ans = kwargs["batch"]["answer"]
        # ensure list of strings
        if isinstance(ans, list):
            return [str(a) for a in ans]
        return [str(ans)]
    if "answer" in kwargs:
        ans = kwargs["answer"]
        if isinstance(ans, list):
            return [str(a) for a in ans]
        return [str(ans)]
    return []

def _extract_text_list_from_completions(completions) -> List[str]:
    """
    TRL/vLLM can hand us a list[List[{'content': str, ...}]] or plain strings.
    We score using the *first* candidate per prompt for GRPO.
    """
    texts = []
    for c in completions:
        if isinstance(c, list) and c:
            # prefer dict with 'content'
            if isinstance(c[0], dict) and "content" in c[0]:
                texts.append(str(c[0]["content"]))
            else:
                texts.append(str(c[0]))
        else:
            texts.append(str(c))
    return texts

def reward_func(prompts, completions, **kwargs):
    print("=====================================================================================================================================")
    print("\n",completions[0],"\n")
    texts = _extract_text_list_from_completions(completions)
    golds = _get_gold_answers_from_kwargs(**kwargs)
    if len(golds) == 0:
        # Fallback: try kwargs['answer'] directly if shapes mismatch
        golds = [""] * len(texts)
    elif len(golds) != len(texts):
        # Shape guard
        if len(golds) == 1:
            golds = golds * len(texts)
        else:
            golds = (golds + [""])[:len(texts)]

    rewards = []
    for txt, gold in zip(texts, golds):
        r = 0.0

        # --- Box enforcement ---
        boxes = list(boxed_re.finditer(txt))
        has_box = len(boxes) >= 1
        if has_box:
            r += 0.6  # presence reward
        else:
            r -= 1.0  # hard penalty if no box at all

        # --- Extract first boxed answer & segments ---
        if has_box:
            b0 = boxes[-1]
            boxed_ans_raw = b0.group(1)
            pre_box_text  = txt[:b0.start()]
            first_box_pos = len(word_tokens(txt[:b0.start()]))  # in tokens
        else:
            boxed_ans_raw = ""
            pre_box_text  = txt
            first_box_pos = -1

        # --- Correctness ---
        if has_box and gold:
            if boxed_ans_raw == gold:
                r += 3.2  # strong correctness reward
            elif boxed_ans_raw.replace(" ", "") == gold.replace(" ", ""):
                r += 1.6

        # --- Real steps before the box ---
        step_score = reasoning_score(pre_box_text)
        r += step_score  # up to ~2.2

        # Slight bonus if there is *some* structure before boxing
        if has_box and (count_equations(pre_box_text) + count_ops(pre_box_text) + count_reasoning_keywords(pre_box_text) >= 2):
            r += 0.3

        # --- Multiple boxes? (Ambiguity) ---
        if len(boxes) > 1:
            r -= 0.5

        # --- Brevity/bloat & answer-first penalties ---
        r += brevity_or_bloat_penalty(txt, pre_box_text, first_box_pos)

        # --- Repetition penalties ---
        r += repetition_penalty(txt)

        # Clamp to a reasonable range for stability
        r = float(max(-3.0, min(6.0, r)))
        rewards.append(r)

    return rewards



# --------------------------
# Optional: LoRA
# --------------------------
def maybe_get_lora():
    if not USE_LORA:
        return None
    try:
        from peft import LoraConfig
    except Exception:
        raise RuntimeError("peft not installed. `pip install peft` or set USE_LORA=0.")
    # Common Qwen2.5 targets (attn + MLP)
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

peft_cfg = maybe_get_lora()
# --------------------------
# GRPO config (training args only)
# --------------------------
N = len(split["train"])                    # number of training examples
WORLD_SIZE = get_world_size()              # of processes/GPUs
PER_DEVICE_BS = BATCH_SIZE                 # per-device microbatch
ACCUM_STEPS = ACCUM                        # gradient_accumulation_steps

grpo_cfg = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    # vllm_gpu_memory_utilization=0.80,
    # vllm_mode="server",
    # vllm_server_base_url="http://127.0.0.1:8010",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    output_dir=OUT_DIR,
    # report_to="tensorboard",
    fp16=True,
    remove_unused_columns=False,  # keep "answer" available in the batch
    max_prompt_length=MAX_PROMPT_TOKENS,
    max_completion_length=MAX_COMPLETION_TOKENS,
    num_generations =GROUP_SIZE,  # valid here (not in GRPOConfig) on 0.21.x
    # use_vllm=True,
    # vllm_mode="server",
    # vllm_server_base_url="http://127.0.0.1:8000",
    # Optional: steer vLLM sampling here (overrides defaults)
    # temperature=0.7,
    # top_p=0.95,
    ddp_timeout=3600,                  # force a raise instead of infinite wait
    dataloader_num_workers=0,          # kill dataloader deadlocks first
    dataloader_pin_memory=False,
)

# --------------------------
# Trainer
# --------------------------
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    args=grpo_cfg,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    reward_funcs=reward_func,
    peft_config=peft_cfg,
)

trainer.train()
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

if USE_LORA:
    merged = trainer.model.merge_and_unload()  # PEFT->base weights
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)

print(f"✅ Done. Saved to: {OUT_DIR}")
