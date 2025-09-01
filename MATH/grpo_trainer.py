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
OUT_DIR    = os.environ.get("OUTPUT_DIR", "../saved_model/out_grpo_qwen_math")

LR         = float(os.environ.get("LR", "5e-6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
ACCUM      = int(os.environ.get("GRAD_ACCUM", "2"))
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "1875"))

MAX_PROMPT_TOKENS     = int(os.environ.get("MAX_PROMPT_TOKENS", "768"))
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "512"))
GROUP_SIZE            = int(os.environ.get("GROUP_SIZE", "4"))  # goes to Trainer (not Config) on 0.21.x

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
    temperature=0.7,
    top_p=0.95,
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

# --------------------------
# Reward function (0.21.x)
# --------------------------
# In TRL 0.21.x, GRPO calls: reward_func(samples: List[str], **kwargs)
# It also forwards the original batch as kwargs["batch"] (dict of columns).
boxed_re = re.compile(r"\\boxed\{([^{}]+)\}")

boxed_re = re.compile(r"\\boxed\{([^{}]+)\}")
def _boxed(s: str) -> str:
    m = boxed_re.search(s.replace(" ", ""))
    return m.group(1) if m else ""

def reward_func(prompts, completions, answer=None, **_):
    # completions is a list of lists of dicts; take the first candidate text
    texts = [c[0]["content"] if isinstance(c, list) and c and "content" in c[0] else str(c) for c in completions]
    golds = [str(a).strip().replace(" ", "") for a in (answer or [""]*len(texts))]
    rewards = []
    for txt, gold in zip(texts, golds):
        r = 0.0
        if _boxed(txt) == gold: r += 2.0               # correctness
        s = txt.lower()
        if ("=" in txt) or ("thus" in s) or ("therefore" in s) or ("hence" in s): r += 1.0  # reasoning signal
        if len(txt.split()) < 12: r -= 0.5             # discourage “just the answer”
        rewards.append(r)
    return rewards

# --------------------------
# GRPO config (training args only)
# --------------------------
grpo_cfg = GRPOConfig(
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
