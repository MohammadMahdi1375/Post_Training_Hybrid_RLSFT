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
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os, json, math, random, re
from typing import Dict, Any, List, Tuple
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# -----------------------------
# Config (override via env)
# -----------------------------
"""
Qwen/Qwen2.5-0.5B
meta-llama/Llama-3.2-1B
Qwen/Qwen2.5-3B
"""
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
MODEL_CONFIG = int(os.environ.get("MODEL_CONFIG", 2))


DATA_DIR     = os.environ.get("DATA_DIR", "/home/mohammad-m/TTT/Post_Training_Hybrid_RLSFT/MATH/data")
TRAIN_FILE   = os.environ.get("TRAIN_FILE", "train_data.json")   # inside DATA_DIR
# TRAIN_FILE   = os.environ.get("TRAIN_FILE", "train_data_easy.json")   # inside DATA_DIR
SPLIT_PCT    = float(os.environ.get("SPLIT_PCT", "0.05"))  # 5% val from train

OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "../saved_model/MATH/qwen_sft_3_512_5e-5")
MERGED_DIR   = os.environ.get("MERGED_DIR", "../saved_model/MATH/qwen_sft_merged_3_512_5e-5")

# training knobs
EPOCHS       = int(os.environ.get("EPOCHS", "3"))
LR           = float(os.environ.get("LR", "5e-5"))
FREEZE_RATE  = float(os.environ.get("FREEZE_RATE", "0.5"))
BSZ          = int(os.environ.get("BSZ", "4"))      # per-device
GR_ACC       = int(os.environ.get("GR_ACC", "8"))  # raise to reach effective batch
MAX_LEN      = int(os.environ.get("MAX_LEN", "512"))
PACKING      = os.environ.get("PACKING", "0") == "1"
WARMUP       = float(os.environ.get("WARMUP", "0.05"))
LOG_STEPS    = int(os.environ.get("LOG_STEPS", "5"))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "10000"))
SEED         = int(os.environ.get("SEED", "42"))

# LoRA knobs
USE_LORA     = os.environ.get("USE_LORA", "1") == "1"
LORA_R       = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))


SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful math tutor. Solve step by step, and put the final answer in \\boxed{...}."
)
# SYSTEM_PROMPT = os.environ.get(
#     "SYSTEM_PROMPT",
#     "You are a helpful math tutor. Read the problem and solve it by givomg reasoning first, then put the final answer in \\boxed{...}."
# )



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
    if USE_LORA:
        os.makedirs(MERGED_DIR, exist_ok=True)

    print(f"[cfg] MODEL_NAME={MODEL_NAME}")
    print(f"[cfg] DATA_DIR={DATA_DIR}")
    print(f"[cfg] OUTPUT_DIR={OUTPUT_DIR}")
    if USE_LORA:
        print(f"[cfg] MERGED_DIR={MERGED_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if MODEL_CONFIG == 2:
        tokenizer.chat_template = """{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif -%}
        {% if message['role'] == 'system' -%}
        <|start_header_id|>system<|end_header_id|>

        {{ message['content'] }}<|eot_id|>
        {% elif message['role'] == 'user' -%}
        <|start_header_id|>user<|end_header_id|>

        {{ message['content'] }}<|eot_id|>
        {% elif message['role'] == 'assistant' -%}
        <|start_header_id|>assistant<|end_header_id|>

        {{ message['content'] if message['content'] is string else '' }}{% if message['content'] is string %}<|eot_id|>{% endif %}
        {% endif %}{% endfor -%}"""
    # Load data
    train_ds_raw, val_ds_raw = load_math_dataset_split_from_train(
        DATA_DIR, TRAIN_FILE, SPLIT_PCT, SEED, OUTPUT_DIR
    )
    print(f"[info] train rows: {len(train_ds_raw)}, val rows: {len(val_ds_raw)}")


    def apply_chat_template(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": example['solution']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}

    # Tokenize the data
    def tokenize_function(example):
        tokenizer.truncation_side = "left"
        tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        # Set padding token labels to -100 to ignore them in loss calculation
        offsets   = tokens["offset_mapping"]
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]

        # marker = "<|im_end|>\n<|im_start|>assistant\n"
        # start_char = example['prompt'].find(marker)

        # if start_char >= 0:
        #     end_char   = start_char + len(marker) if start_char != -1 else -1
        #     for i, (a, b) in enumerate(offsets):
        #         if a == b:  # special tokens
        #             continue
        #         if not (b > end_char and a < (len(example["prompt"]) - len(marker))):
        #             tokens['labels'][i] = -100
        return tokens


    train_ds = train_ds_raw.map(
        apply_chat_template,
        remove_columns=[c for c in train_ds_raw.column_names if c not in ("text","ans_boxed")]
    )
    val_ds = val_ds_raw.map(
        apply_chat_template,
        remove_columns=[c for c in val_ds_raw.column_names if c not in ("text","ans_boxed")]
    )

    train_ds = train_ds.map(tokenize_function)
    val_ds   = val_ds.map(tokenize_function)





    # Collator that masks everything before the assistant span

    print("[info] Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # torch_dtype=torch.float16,
        device_map="auto",  # let accelerate place for QLoRA; for LoRA fp16 we let Trainer handle DDP
        trust_remote_code=True,
    )

    # PEFT LoRA
    if USE_LORA:
        peft_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "up_proj","down_proj","gate_proj",
            ],
        )
        model = get_peft_model(model, peft_cfg)
    else:
        lora = None
        blocks = model.model.layers
        cutt_off = int(FREEZE_RATE * len(blocks))

        for i, block in enumerate(blocks):
            if i < cutt_off:
                for p in block.parameters():
                    p.requires_grad = False



    # SFT config — NOTE: no 'train_on_source' or 'add_generation_prompt' (those caused your earlier errors)
    model.train()
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps", # To evaluate during training
        eval_steps=10000,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        per_device_train_batch_size=BSZ, # Adjust based on your hardware
        num_train_epochs=EPOCHS, # How many times to loop through the dataset
        warmup_ratio=WARMUP,
        # fp16=True, # Must be False for MacBooks
        report_to="none", # Here we can use something like tensorboard to see the training metrics
        log_level="info",
        learning_rate=LR, # Would avoid larger values here
        max_grad_norm=GR_ACC # Clipping the gradients is always a good idea
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
        )

# Train the model

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"[sanity] Trainable params (LoRA): {trainable:,} / {total:,}")

    print("[info] Starting training ...")
    trainer.train()

    

    # Merge LoRA into base weights to get a single, vLLM-ready model dir
    print("[info] Merging LoRA and saving merged model ...")
    merged = trainer.model.merge_and_unload()  # PEFT->base weights
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Adapter (for LoRA serving): {MERGED_DIR}")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Finetuned model: {OUTPUT_DIR}")


    # Small sanity print
    print("\n=== Config ===")
    pairs = [
        ("MODEL_NAME", MODEL_NAME),
        ("TRAIN_FILE", TRAIN_FILE),
        ("OUTPUT_DIR", OUTPUT_DIR),
        ("MERGED_DIR", MERGED_DIR),
        ("EPOCHS", EPOCHS),
        ("LR", LR),
        ("FREEZE_RATE", FREEZE_RATE),
        ("BSZ", BSZ),
        ("GR_ACC", GR_ACC),
        ("MAX_LEN", MAX_LEN),
        ("PACKING", PACKING),
        ("WARMUP", WARMUP),
        ("LOG_STEPS", LOG_STEPS),
        ("USE_LORA", USE_LORA),
        ("LORA_R", LORA_R),
        ("LORA_ALPHA", LORA_ALPHA),
        ("EFFECTIVE_BATCH", BSZ * GR_ACC),
    ]
    w = max(len(k) for k, _ in pairs)
    for k, v in pairs:
        print(f"{k:<{w}} = {v}")

    

if __name__ == "__main__":
    main()
