#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-Training (SFT) for Qwen/Qwen2.5-0.5B-Instruct on MATH-style data.

- TRL 0.21.0, Transformers 4.55.4, PEFT >= 0.10
- V100-friendly (fp16). Optional QLoRA path included.
- Trains with completion-only loss on the assistant span using DataCollatorForCompletionOnlyLM.
- Builds prompts with Qwen chat template; trains on 'solution' (full reasoning with boxed answer).
- Saves BOTH the LoRA adapter AND a merged full model (ready for your vLLM evaluator).

Env vars you may override:
  MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
  DATA_DIR=/path/to/data
  TRAIN_FILE=train_data.json
  VAL_FILE=val_data.json   (optional; if missing, we split from train)
  OUTPUT_DIR=/path/to/out/qwen25_05b_sft_model
  MERGED_DIR=/path/to/out/merged
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import os, json, math, random, re
from typing import Dict, Any, List, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# -----------------------------
# Config (override via env)
# -----------------------------
MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

DATA_DIR    = os.environ.get("DATA_DIR", "/home/mohammad-m/TTT/Post_Training_Hybrid_RLSFT/MATH/data")
TRAIN_FILE  = os.environ.get("TRAIN_FILE", "train_data.json")   # inside DATA_DIR
SPLIT_PCT = float(os.environ.get("SPLIT_PCT", "0.05"))  # 5% val from train

OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/home/mohammad-m/TTT/saved_model/MATH/sft_FT_1")

# training knobs
EPOCHS      = int(os.environ.get("EPOCHS", "1"))
LR          = float(os.environ.get("LR", "1e-7"))
FREEZE_RATE = float(os.environ.get("FREEZE_RATE", "0.0"))
BSZ         = int(os.environ.get("BSZ", "4"))      # per-device
GR_ACC      = int(os.environ.get("GR_ACC", "16"))  # raise to reach effective batch
MAX_LEN     = int(os.environ.get("MAX_LEN", "1024"))
PACKING     = os.environ.get("PACKING", "0") == "1"
WARMUP      = float(os.environ.get("WARMUP", "0.05"))
LOG_STEPS   = int(os.environ.get("LOG_STEPS", "5"))
SAVE_STEPS  = int(os.environ.get("SAVE_STEPS", "1000"))
SEED        = int(os.environ.get("SEED", "42"))

# LoRA knobs
LORA_R      = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA  = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT= float(os.environ.get("LORA_DROPOUT", "0.05"))


SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful math tutor. Solve step by step, and put the final answer in \\boxed{...}."
)
BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")

def set_seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _read_json_file(path: str) -> List[Dict[str, Any]]:
    """
    Robust reader:
      - JSON array:            [ {...}, {...}, ... ]
      - Single JSON object:    {...}
      - JSONL:                 {...}\n{...}\n...
      - Handles UTF-8 BOM and blank lines.
    """
    def _valid(x):
        return isinstance(x, dict) and ("problem" in x) and ("solution" in x or "answer" in x)

    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try: JSON array / object first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if _valid(r)]
        elif isinstance(obj, dict):
            return [obj] if _valid(obj) else []
    except json.JSONDecodeError:
        pass  # fall back to line-by-line

    # Fallback: JSONL (one object per non-empty line)
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if _valid(rec):
                out.append(rec)
        except json.JSONDecodeError as e:
            # Helpful context; you can keep this as print or raise
            print(f"[warn] Skipping line {i} in {path}: {e}")
            continue
    return out


def load_math_dataset_split_from_train(
        data_dir: str,
        train_file: str,
        split_pct: float,
        seed: int,
        output_dir: str,
    ) -> tuple[Dataset, Dataset]:
        """
        Read *only* train_data.json (or .jsonl), then create a reproducible split:
        - val = round(split_pct * N), min 1
        - indices saved to <output_dir>/split_indices.json so subsequent runs reuse it
        """
        train_path = os.path.join(data_dir, train_file)
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")

        # robust reader you already have (JSON / JSONL tolerant)
        train_list: list[dict[str, Any]] = _read_json_file(train_path)
        n = len(train_list)
        if n == 0:
            raise ValueError(f"No valid records found in {train_path}.")

        os.makedirs(output_dir, exist_ok=True)
        split_file = os.path.join(output_dir, "split_indices.json")

        # Reuse the same split if it exists
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_meta = json.load(f)
            val_idx = split_meta.get("val_indices", [])
            if not val_idx:
                raise ValueError(f"{split_file} exists but has no 'val_indices'. Delete it to re-split.")
            print(f"[info] Reusing validation split from {split_file} ({len(val_idx)} rows)")
        else:
            rnd = random.Random(seed)
            indices = list(range(n))
            rnd.shuffle(indices)
            k = max(1, int(round(split_pct * n)))
            val_idx = sorted(indices[:k])
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": os.path.abspath(train_path),
                        "num_rows": n,
                        "split_pct": split_pct,
                        "seed": seed,
                        "val_indices": val_idx,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[info] Created validation split: {k} val / {n-k} train (saved -> {split_file})")

        # Materialize lists
        val_set  = [train_list[i] for i in val_idx]
        val_mask = set(val_idx)
        trn_set  = [r for i, r in enumerate(train_list) if i not in val_mask]

        # Return HF datasets
        return Dataset.from_list(trn_set), Dataset.from_list(val_set)



def get_boxed_answer(solution: str, answer: str | None) -> str | None:
    matches = list(BOXED_RE.finditer(solution))
    if matches:
        val = matches[-1].group(1).strip()
        return f"\\boxed{{{val}}}"
    if answer is not None:
        return f"\\boxed{{{str(answer).strip()}}}"
    return None



def main():
    set_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[cfg] MODEL_NAME={MODEL_NAME}")
    print(f"[cfg] DATA_DIR={DATA_DIR}")
    print(f"[cfg] OUTPUT_DIR={OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_ds_raw, val_ds_raw = load_math_dataset_split_from_train(
        DATA_DIR, TRAIN_FILE, SPLIT_PCT, SEED, OUTPUT_DIR
    )
    print(f"[info] train rows: {len(train_ds_raw)}, val rows: {len(val_ds_raw)}")

    # Map to a single 'text' field that contains full chat-rendered dialogue
    def to_text(example):
        problem  = example.get("problem", "").strip()
        solution = example.get("solution", example.get("answer", "")).strip()
        if not solution and "answer" in example:
            solution = str(example["answer"]).strip()

        boxed = get_boxed_answer(solution, example.get("answer"))

        assistant_text = solution
        if boxed and boxed not in solution:
            assistant_text = solution + f"\n\nFinal Answer: {boxed}"
        elif boxed:
            assistant_text = solution + f"\n\nFinal Answer: {boxed}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve the following problem:\n\n{problem}"},
            {"role": "assistant", "content": assistant_text},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": rendered, "ans_boxed": boxed or ""}


    train_ds = train_ds_raw.map(
        to_text,
        remove_columns=[c for c in train_ds_raw.column_names if c not in ("text","ans_boxed")]
    )
    val_ds = val_ds_raw.map(
        to_text,
        remove_columns=[c for c in val_ds_raw.column_names if c not in ("text","ans_boxed")]
    )

    def tokenize_and_mask(ex):
        txt   = ex["text"]
        boxed = ex.get("ans_boxed", "")
        old = tokenizer.truncation_side
        tokenizer.truncation_side = "left"   # keep the end (answers are at end)
        out = tokenizer(txt, truncation=True, max_length=MAX_LEN,
                        padding=False, return_offsets_mapping=True)
        tokenizer.truncation_side = old

        input_ids = out["input_ids"]
        attn      = out["attention_mask"]
        offsets   = out["offset_mapping"]
        labels    = [-100] * len(input_ids)

        if boxed:
            start = txt.rfind(boxed)
            if start >= 0:
                end = start + len(boxed)
                for i, (a, b) in enumerate(offsets):
                    if a == b:  # special tokens
                        continue
                    if not (b <= start or a >= end):
                        labels[i] = input_ids[i]

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    train_ds = train_ds.map(tokenize_and_mask, remove_columns=["text","ans_boxed"])
    val_ds   = val_ds.map(tokenize_and_mask,   remove_columns=["text","ans_boxed"])

    # (optional) ensure tensor outputs
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])





    # Collator that masks everything before the assistant span

    print("[info] Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # torch_dtype=torch.float16,
        device_map="auto",  # let accelerate place for QLoRA; for LoRA fp16 we let Trainer handle DDP
        trust_remote_code=True,
    )

    print(model)
    blocks = model.model.layers
    cutt_off = int(FREEZE_RATE * len(blocks))

    for i, block in enumerate(blocks):
        if i < cutt_off:
            for p in block.parameters():
                p.requires_grad = False



    # SFT config — NOTE: no 'train_on_source' or 'add_generation_prompt' (those caused your earlier errors)
    # SFT config: we provide our own labels, so no completion_only_loss
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BSZ,
        gradient_accumulation_steps=GR_ACC,
        max_length=MAX_LEN,
        packing=False,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True, bf16=False,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=max(SAVE_STEPS, LOG_STEPS * 10),
        load_best_model_at_end=True,
        report_to="none",
        completion_only_loss=False,          # we already set labels
        remove_unused_columns=False,         # <-- keep labels in the batch
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )


    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"[sanity] Trainable params (LoRA): {trainable:,} / {total:,}")

    print("[info] Starting training ...")
    trainer.train()

    print("[info] Saving adapter ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)



    # Small sanity print
    print("\n=== Done ===")
    print(f"Adapter (for LoRA serving): {OUTPUT_DIR}")

if __name__ == "__main__":
    main()