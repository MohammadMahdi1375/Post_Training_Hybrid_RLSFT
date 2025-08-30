#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Answer-only SFT for Qwen/Qwen2.5-0.5B-Instruct on EleutherAI/hendrycks_math (all subjects)
# - Loss is computed ONLY on the final answer string.
# - Keeps model’s internal reasoning intact; you can still prompt for steps at inference.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from typing import List, Dict, Any
import torch
import os, re, math, torch
from typing import List, Dict
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, default_data_collator)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/mohammad-m/TTT/saved_model/MATH/qwen25_math_ansonly_sft")
MERGED_DIR  = os.environ.get("MERGED_DIR", "/home/mohammad-m/TTT/saved_model/MATH/merged_1")
USE_4BIT   = os.environ.get("USE_4BIT", "0") == "1"
MAX_LEN    = int(os.environ.get("MAX_LEN", "1024"))  # answer-only needs far less context
LR         = float(os.environ.get("LR", "2e-4"))
EPOCHS     = float(os.environ.get("EPOCHS", "1"))
BSZ        = int(os.environ.get("BSZ", "8"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", "4"))
SEED       = int(os.environ.get("SEED", "42"))
VAL_FRACT  = float(os.environ.get("VAL_FRACT", "0.02"))

SUBJECTS = ["algebra","counting_and_probability","geometry","intermediate_algebra",
            "number_theory","prealgebra","precalculus"]

def load_all_subjects(split: str):
    parts = [load_dataset("EleutherAI/hendrycks_math", cfg, split=split) for cfg in SUBJECTS]
    return concatenate_datasets(parts)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
def extract_final_answer(example):
    # Prefer explicit 'answer'; else try to pull \boxed{...} from 'solution'.
    ans = (example.get("answer") or "").strip()
    if ans:
        return ans
    sol = (example.get("solution") or "").strip()
    m = BOXED_RE.search(sol)
    return (m.group(1).strip() if m else sol.splitlines()[-1].strip()) if sol else ""

def build_prompt(problem: str) -> str:
    return (f"Solve the following problem:\n\n{problem}")

def tokenize_answer_only(tok, problem: str, answer: str, max_len: int) -> Dict[str, torch.Tensor]:
    # 1) Build chat prompt with empty assistant to include assistant prefix tokens
    messages = [
        {"role":"system","content":"You are a helpful math tutor. Solve step by step, and put the final answer in \\boxed{...}."},
        {"role":"user","content": build_prompt(problem)},
        {"role":"assistant","content": ""},  # assistant prefix only
    ]
    prompt_ids: List[int] = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    # 2) Target string: short, canonicalized final answer
    # Encourage consistent format during training:
    target_text = f"\\boxed{{{answer}}}"
    target_ids: List[int] = tok(target_text, add_special_tokens=False).input_ids
    eos_id = tok.eos_token_id
    if not target_ids or target_ids[-1] != eos_id:
        target_ids.append(eos_id)

    input_ids = prompt_ids + target_ids
    labels    = [-100]*len(prompt_ids) + target_ids
    attn      = [1]*len(input_ids)

    # Left-truncate the prompt first (preserve targets)
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        cut = min(overflow, len(prompt_ids))
        input_ids = input_ids[cut:]
        labels    = labels[cut:]
        attn      = attn[cut:]
        overflow -= cut
        if overflow > 0:  # extremely unlikely for short answers
            input_ids = input_ids[overflow:]; labels = labels[overflow:]; attn = attn[overflow:]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

def preprocess(tok, ds, max_len):
    def _map(ex):
        problem = ex.get("problem","")
        answer = extract_final_answer(ex)
        return tokenize_answer_only(tok, problem, answer, max_len)
    cols = list(set(ds.column_names) - {"input_ids","labels","attention_mask"})
    return ds.map(_map, remove_columns=cols)




@dataclass
class DataCollatorForCausalLM:
    tokenizer: any
    pad_to_multiple_of: int | None = 8   # good for tensor cores on V100; set None to disable

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Ensure everything is tensors
        ids_list  = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_list = [torch.as_tensor(f["attention_mask"], dtype=torch.long) for f in features]
        lab_list  = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in ids_list)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            if max_len % m:
                max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        bsz = len(features)
        input_ids     = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attention_mask= torch.zeros((bsz, max_len), dtype=torch.long)
        labels        = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, (ids, attn, lab) in enumerate(zip(ids_list, attn_list, lab_list)):
            L = ids.size(0)
            input_ids[i, :L]      = ids
            attention_mask[i, :L] = attn
            labels[i, :L]         = lab

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    train_all = load_all_subjects("train")
    train_all = train_all.train_test_split(test_size=VAL_FRACT, seed=SEED)
    train_ds, val_ds = train_all["train"], train_all["test"]

    train_tok = preprocess(tok, train_ds, MAX_LEN)
    val_tok   = preprocess(tok, val_ds, MAX_LEN)

    # Model (QLoRA optional)
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=qcfg,
                                                     device_map="auto", torch_dtype=torch.float16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_tok) / (BSZ * max(1, torch.cuda.device_count()) * GRAD_ACC))
    collator = DataCollatorForCausalLM(tokenizer=tok, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BSZ,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(0.03 * steps_per_epoch * EPOCHS)),
        logging_steps=max(5, steps_per_epoch // 50),
        save_steps=max(100, steps_per_epoch // 2),
        eval_strategy="steps",
        eval_steps=max(100, steps_per_epoch // 2),
        save_total_limit=2,
        fp16=True,
        weight_decay=0.05,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        data_collator=collator,        # <-- instead of default_data_collator
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    print("[info] Merging LoRA and saving merged model ...")
    merged = trainer.model.merge_and_unload()  # PEFT->base weights
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)

if __name__ == "__main__":
    main()
