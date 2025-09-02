#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO Post-Training for Qwen/Qwen2.5-0.5B-Instruct on MATH data.

- Uses TRL 0.21.0, Transformers 4.55.4, PEFT >= 0.10
- V100 friendly (fp16), LoRA optional.
- Dataset format: JSON with keys: idx, problem, solution, answer
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # avoid CUDA+fork issues


import os, json, re
from typing import List, Dict, Any
import math, os, torch
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
OUT_DIR    = os.environ.get("OUTPUT_DIR", "../saved_model/MATH/out_grpo_qwen_math")

LR         = float(os.environ.get("LR", "5e-6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
ACCUM      = int(os.environ.get("GRAD_ACCUM", "2"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "3"))

MAX_PROMPT_TOKENS     = int(os.environ.get("MAX_PROMPT_TOKENS", "768"))
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "1024"))
GROUP_SIZE            = int(os.environ.get("GROUP_SIZE", "4"))  # goes to Trainer (not Config) on 0.21.x







W_CORRECT       = float(os.environ.get("W_CORRECT", "2.0"))   # Final Answer Correctness
W_BOX_COMPLY_Y  = float(os.environ.get("W_BOX_COMPLY_Y", "0.5"))  # Boxed present
W_BOX_COMPLY_N  = float(os.environ.get("W_BOX_COMPLY_N", "0.5"))  # Boxed missing (penalty)
W_LEN           = float(os.environ.get("W_LEN", "1.0"))       # Length shaping scale
LEN_THRESHOLD   = int(os.environ.get("LEN_THRESHOLD", "64"))  # token threshold (>= encouraged)

W_REASON        = float(os.environ.get("W_REASON", "1.0"))    # Intermediate reasoning shaping
REASON_MAX_EQ   = int(os.environ.get("REASON_MAX_EQ", "6"))   # cap for '='/markers density



# --------------------------
# Load model / tokenizer
# --------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype=torch.float16,
    device_map="auto",
)

model.generation_config = GenerationConfig(
    do_sample=True,     # GRPO benefits from sampling
    temperature=1,
    top_p=0.95,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)

# --------------------------
# Data loading/helpers
# --------------------------
def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    # Accelerate sets WORLD_SIZE; fall back to 1
    return int(os.environ.get("WORLD_SIZE", "1"))



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
    problem  = str(ex["problem"]).strip()
    answer   = str(ex.get("answer", "")).strip()
    solution = str(ex.get("solution", "")).strip()  # <-- keep ref reasoning
    rows.append({
        "prompt": build_prompt(problem),
        "answer": answer,
        "solution": solution,
    })

ds_all = Dataset.from_list(rows)
split = ds_all.train_test_split(test_size=0.05, seed=42)

# --------------------------
# Reward function (0.21.x)
# --------------------------
# In TRL 0.21.x, GRPO calls: reward_func(samples: List[str], **kwargs)
# It also forwards the original batch as kwargs["batch"] (dict of columns).
boxed_re = re.compile(r"\\boxed\{([^{}]+)\}")

def _boxed_last(s: str) -> str:
    # use the **last** boxed expression, spaces ignored
    matches = boxed_re.findall(s.replace(" ", ""))
    return matches[-1] if matches else ""

_reason_markers = ("thus", "therefore", "hence", "so", "then", "we obtain", "we get", "implies")

def _clean_math_text(s: str) -> str:
    s = s.lower()
    # strip some LaTeX control noise but keep math-y chars
    s = re.sub(r"\\[a-zA-Z]+", " ", s)           # \text, \frac, \begin, ...
    s = re.sub(r"[^a-z0-9=+\-*/^()., ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str):
    # words, numbers, and individual math symbols as "tokens"
    return set(re.findall(r"[a-z]+|\d+|[=+\-*/^()]", s))

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

def _seq_ratio(a: str, b: str) -> float:
    # difflib gives a 0..1 rough relevance signal
    try:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def reward_func(prompts, completions, answer=None, solution=None, **_):
    # completions: list of lists of dicts (first candidate is used by TRL per-sample);
    # we'll score **each** candidate list by its first element's 'content'.
    texts = [
        (c[0]["content"] if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]
         else (c[0] if isinstance(c, list) and c else str(c)))
        for c in completions
    ]

    golds      = [str(a or "").strip().replace(" ", "") for a in (answer or [""] * len(texts))]
    ref_sols   = [str(s or "") for s in (solution or [""] * len(texts))]

    rewards = []
    for txt, gold, ref_sol in zip(texts, golds, ref_sols):
        r = 0.0

        # ---------- (1) Final Answer Correctness ----------
        boxed_val = _boxed_last(txt)
        if boxed_val == gold:
            r += W_CORRECT

        # ---------- (2) Boxed-Format Compliance ----------
        has_box = bool(boxed_re.search(txt))
        r += (W_BOX_COMPLY_Y if has_box else -W_BOX_COMPLY_N)

        # ---------- (3) Length Shaping (tokens) ----------
        # Encourage >= LEN_THRESHOLD tokens; penalize shorter
        # (uses tokenizer already loaded as `tok`)
        try:
            n_tokens = len(tok.encode(txt))
        except Exception:
            n_tokens = len(txt.split())
        r += (W_LEN if n_tokens >= LEN_THRESHOLD else -W_LEN)

        # ---------- (4) Intermediate Reasoning Shaping ----------
        # Relevance/matching to the provided step-by-step solution.
        # Combines: (a) difflib similarity, (b) Jaccard over mathy tokens,
        # and (c) density of equations/markers (structure signal).
        clean_gen = _clean_math_text(txt)
        clean_ref = _clean_math_text(ref_sol)

        seq_sim   = _seq_ratio(clean_gen, clean_ref)            # 0..1
        jac_sim   = _jaccard(_token_set(clean_gen), _token_set(clean_ref))  # 0..1

        s_lower   = clean_gen
        marker_cnt = sum(m in s_lower for m in _reason_markers)
        eq_cnt     = txt.count("=")
        struct_density = min((marker_cnt + eq_cnt) / max(REASON_MAX_EQ, 1), 1.0)  # 0..1

        # Weighted blend → 0..1 (center later to allow positive/negative shaping)
        relevance = 0.5 * seq_sim + 0.3 * jac_sim + 0.2 * struct_density

        # Center to [-1, +1] then scale → encourages match, discourages irrelevance
        r += W_REASON * (2.0 * relevance - 1.0)

        rewards.append(r)

    return rewards


# --------------------------
# GRPO config (training args only)
# --------------------------
N = len(split["train"])                    # number of training examples
WORLD_SIZE = get_world_size()              # # of processes/GPUs
PER_DEVICE_BS = BATCH_SIZE                 # per-device microbatch
ACCUM_STEPS = ACCUM                        # gradient_accumulation_steps

# HF Trainer logic in a nutshell:
# len(train_dataloader) ≈ ceil(N / (PER_DEVICE_BS * WORLD_SIZE))
microsteps_per_epoch = math.ceil(N / (PER_DEVICE_BS * WORLD_SIZE))
steps_per_epoch = math.ceil(microsteps_per_epoch / ACCUM_STEPS)

MAX_STEPS = int(NUM_EPOCHS * steps_per_epoch)


grpo_cfg = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUM,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=500,
    output_dir=OUT_DIR,
    report_to="tensorboard",
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
    # IMPORTANT in TRL 0.21.x:
    # - No `formatting_func` here.
    # - Provide generation and length controls directly to the trainer:
    # Column used as the input prompt:
)

trainer.train()
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

print(f"✅ Done. Saved to: {OUT_DIR}")
