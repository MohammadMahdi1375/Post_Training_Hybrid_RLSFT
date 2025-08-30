#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Prompt–completion SFT for Qwen/Qwen2.5-0.5B-Instruct (TRL 0.21.x, HF 4.55.4)
# Dataset rows: {"problem": str, "solution": str[, "level": str/int, "type": str]}
# Produces {"prompt": <chat-prompt>, "completion": <assistant text ending with <|im_end|>>}

import os, json, glob
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# -----------------------------
# Config (override via env)
# -----------------------------
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR     = os.environ.get("DATA_DIR", "/home/mohammad-m/TTT/Post_Training_Hybrid_RLSFT/MATH/data")
TRAIN_FILE   = os.environ.get("TRAIN_FILE", "train_data.json")  # inside DATA_DIR
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "/home/mohammad-m/TTT/saved_model/MATH/qwen25_05b_sft_lora_prompt_completion")

MAX_LEN      = int(os.environ.get("MAX_SEQ_LEN", 2048))
TRAIN_BS     = int(os.environ.get("TRAIN_BS", 4))
EVAL_BS      = int(os.environ.get("EVAL_BS", 1))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", 16))
EPOCHS       = float(os.environ.get("EPOCHS", 1))
LR           = float(os.environ.get("LR", 1e-5))
WARMUP       = float(os.environ.get("WARMUP_RATIO", 0.03))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
LOG_STEPS    = int(os.environ.get("LOG_STEPS", 25))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", 500))
EVAL_STEPS   = int(os.environ.get("EVAL_STEPS", 500))

BF16         = os.environ.get("BF16", "0") == "1"   # keep False on V100
PACKING      = os.environ.get("PACKING", "0") == "1"

# Optional filtering
FILTER_LEVELS = os.environ.get("FILTER_LEVELS")  # e.g. "1,2,3"
ONLY_EASY     = os.environ.get("ONLY_EASY", "0") == "1"   # < 4
ONLY_HARD     = os.environ.get("ONLY_HARD", "0") == "1"   # >= 4
DEV_SPLIT_PCT = float(os.environ.get("DEV_SPLIT_PCT", 0.05))  # last 5% as eval

# LoRA / QLoRA
USE_LORA     = os.environ.get("USE_LORA", "1") == "1"
LOAD_4BIT    = os.environ.get("LOAD_4BIT", "0") == "1"
LORA_R       = int(os.environ.get("LORA_R", 32))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", 64))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
LORA_TARGET  = os.environ.get(
    "LORA_TARGET", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

SYSTEM_PROMPT = "You are a helpful math tutor. Solve step by step, and put the final answer in \\boxed{...}."

# -----------------------------
# Helpers
# -----------------------------
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        # naive detection of jsonl
        if text.strip().startswith("{") and "\n{" in text:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        else:
            obj = json.loads(text)
            if isinstance(obj, dict): out.append(obj)
            elif isinstance(obj, list): out.extend(obj)
            else: raise ValueError(f"Unsupported JSON in {path}")
    return out

def level_to_int(level_str: Any) -> int:
    try:
        s = str(level_str).strip()
        # handle "Level 3" or "3"
        toks = s.split()
        return int(toks[-1]) if toks else 0
    except Exception:
        return 0

def apply_filters(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if FILTER_LEVELS:
        keep = {int(x) for x in FILTER_LEVELS.split(",")}
        rows = [r for r in rows if level_to_int(r.get("level", "")) in keep]
    if ONLY_EASY:
        rows = [r for r in rows if level_to_int(r.get("level", "")) < 4]
    if ONLY_HARD:
        rows = [r for r in rows if level_to_int(r.get("level", "")) >= 4]
    return rows

def load_rows(data_dir: str, filename: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(data_dir, filename)))
    if not files:
        raise FileNotFoundError(f"No file '{filename}' found in {data_dir}")
    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            for r in read_json_or_jsonl(fp):
                if "problem" in r and ("solution" in r or "answer" in r):
                    if "solution" not in r and "answer" in r:
                        r["solution"] = str(r["answer"])
                    rows.append(r)
        except Exception as e:
            print(f"[warn] skipping {fp}: {e}")
    print(f"Loaded {len(rows)} examples from {len(files)} file(s).")
    return rows

def to_prompt_completion(tokenizer, problem: str, solution: str) -> Dict[str, Any]:
    """
    Create a prompt that ends at the start of the assistant turn and
    a completion that contains ONLY the assistant text, ending with <|im_end|>.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Solve the following problem:\n\n{problem}"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True  # leaves "...<|assistant|>\n"
    )

    # Ensure completion ends with Qwen's end-of-turn token.
    im_end = "<|im_end|>"
    completion = solution.strip()
    if not completion.endswith(im_end):
        completion = completion + im_end

    return {"prompt": prompt_text, "completion": completion}



# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("Loading data + filters...")
    rows = apply_filters(load_rows(DATA_DIR, TRAIN_FILE))

    # split (last DEV_SPLIT_PCT as eval)
    n = len(rows)
    split = max(1, int((1.0 - DEV_SPLIT_PCT) * n))
    train_rows, eval_rows = rows[:split], rows[split:]

    print("Rendering prompt–completion pairs...")
    def pack(rr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [to_prompt_completion(tok, r["problem"], r["solution"]) for r in rr]

    train_list = pack(train_rows)
    eval_list  = pack(eval_rows)



    train_ds = Dataset.from_list(train_list)
    eval_ds  = Dataset.from_list(eval_list) if len(eval_list) else None

    print("Loading model...")
    extra_kwargs = {}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=None,  # V100-safe
        **extra_kwargs
    )

    # LoRA (optional)
    peft_config = None
    if USE_LORA:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET,
            task_type="CAUSAL_LM",
            bias="none",
        )
    else:
        model.requires_grad_(True)
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: [{trainable}/{total}]")

    # --- TRL 0.21 SFT ---
    # We pass a prompt–completion dataset and DO NOT provide a formatting function.
    # This keeps it compatible with completion_only_loss=True.
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOG_STEPS,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=BF16,
        fp16=(not BF16),
        dataloader_num_workers=2,
        report_to=[],
        optim="adamw_torch",
        max_grad_norm=1.0,
        max_length=MAX_LEN,
        packing=PACKING,
        padding_free=False,            # safer on V100 when packing
        completion_only_loss=True,     # <-- train only on completions
        # NOTE: Don't set assistant_only_loss for prompt–completion datasets.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,   # tokenizer
        peft_config=peft_config,
        # No formatting function, no collator override — TRL picks prompt–completion path.
    )

    print("Training...")
    trainer.train()

    print("Saving...")
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # quick sanity generation
    demo_prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Compute: 3^4 - 5\\cdot 8."}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    demo_inputs = tok(demo_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**demo_inputs, max_new_tokens=128, do_sample=False)
    print("\n=== SAMPLE OUTPUT ===")
    print(tok.decode(out[0], skip_special_tokens=False))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
