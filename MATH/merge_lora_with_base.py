from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

BASE = "Qwen/Qwen2.5-0.5B-Instruct"              # your base
ADAPTER = "/home/mohammad-m/TTT/Post-Training/MATH/saved_model/qwen25_05b_sft_lora_prompt_completion"                     # your trained LoRA
OUT = "/home/mohammad-m/TTT/Post-Training/MATH/saved_model/merged/"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="float16", trust_remote_code=True, device_map="cpu")
merged = PeftModel.from_pretrained(base, ADAPTER)
merged = merged.merge_and_unload()               # bake LoRA into the backbone

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)  # writes safetensors
tok.save_pretrained(OUT)