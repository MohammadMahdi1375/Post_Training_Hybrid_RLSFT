import os, re, json, glob, math
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Config via env ----
os.environ.get['CUDA_VISIBLE_DEVICES'] = "6, 7"
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR   = os.environ.get("DATA_DIR", "/home/mohammad-m/TTT/Post-Training/MATH/data")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
TOP_P = float(os.environ.get("TOP_P", 1.0))
LIMIT = os.environ.get("LIMIT")  # e.g., "500"
FILTER_LEVELS = os.environ.get("FILTER_LEVELS")  # e.g., "1,2,3"
ONLY_EASY     = os.environ.get("ONLY_EASY", "0") == "1"
ONLY_HARD     = os.environ.get("ONLY_HARD", "0") == "1"

SYSTEM_PROMPT = "You are a helpful math tutor. Solve step by step, and put the final answer in \\boxed{...}."


def read_json_or_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()          # 🔹 read ALL characters
        f.seek(0)                   # reset pointer (optional, in case we want to reuse f)

        # If file looks like JSONL (multiple JSON objects, one per line)
        if "\n" in content and content.strip().startswith("{") and "\n{" in content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        else:
            # Normal JSON (either a dict or a list of dicts)
            obj = json.loads(content)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend(obj)
            else:
                raise ValueError(f"Unsupported JSON in {path}")
    return out

def load_rows(data_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(data_dir, "test_data.json"), recursive=True))
    rows = []
    for fp in files:
        try:
            for r in read_json_or_jsonl(fp):
                if "problem" in r and "solution" in r:
                    rows.append(r)
        except Exception as e:
            print(f"[warn] skipping {fp}: {e}")
    print(f"Loaded {len(rows)} examples from {len(files)} files.")
    return rows

def level_to_int(level_str: str) -> int:
    try:
        return int(str(level_str).strip().split()[-1])
    except Exception:
        return 0

def apply_filters(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if FILTER_LEVELS:
        keep = {int(x) for x in FILTER_LEVELS.split(",")}
        rows = [r for r in rows if level_to_int(r.get("level","")) in keep]
    if ONLY_EASY:
        rows = [r for r in rows if level_to_int(r.get("level","")) < 4]
    if ONLY_HARD:
        rows = [r for r in rows if level_to_int(r.get("level","")) >= 4]
    if LIMIT:
        rows = rows[:int(LIMIT)]
    return rows

def normalize_tex(s: str) -> str:
    s = s.strip()   ##### remove leading/trailing spaces.
    s = re.sub(r"\\boxed\s*\{([^{}]|{[^{}]*})*\}", lambda m: m.group(0)[7:-1], s)   ##### if the answer is wrapped in \boxed{...}, strip off the \boxed{} wrapper. For example: \boxed{41} → 41
    s = s.replace("$$", "$").strip("$ ").replace(" ", "").replace(",", "")  ##### clean display-math markers ($$), outer $...$, spaces, commas: For example: "$41$" → "41" and "1,000" → "1000"
    s = re.sub(r"\\left|\\right", "", s)    ##### remove LaTeX size-adjusters (\left, \right). For example: \left(1/2\right) → (1/2)
    s = re.sub(r"\\(d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\2)/(\3)", s)    ##### convert LaTeX fractions (\frac{a}{b} or \dfrac{a}{b}) into plain (a)/(b). For example: \frac{3}{4} → (3)/(4)
    if s.endswith("."): s = s[:-1]  ##### drop trailing periods (so "41." matches "41").
    return s

def extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\s*\{([^{}]|{[^{}]*})*\}", text, flags=re.DOTALL)
    if boxed:
        last = boxed[-1]
        last = re.sub(r"^\{(.*)\}$", r"\1", last.strip())
        return last.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"([=\:]\s*)?([-+]?[\d]+(\.[\d]+)?)$", ln)
        if m: return m.group(2)
        m2 = re.search(r"([-+]?\d+)\s*/\s*([-+]?\d+)$", ln)
        if m2: return f"{m2.group(1)}/{m2.group(2)}"
    tail = re.findall(r"[A-Za-z0-9\\\(\)\+\-\*/^_=\.]+", text)
    return tail[-1] if tail else text.strip()

def approx_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    na, nb = normalize_tex(a), normalize_tex(b)
    if na == nb: return True
    ##### Try to parse each string as a number: If it looks like a fraction ("3/4") and has no letters, turn into 0.75. Otherwise, try to parse as float(x)
    def to_float(x: str) -> Optional[float]:
        try:
            if "/" in x and not any(c.isalpha() for c in x):
                p, q = x.split("/", 1); return float(p)/float(q)
            return float(x)
        except Exception: return None
    fa, fb = to_float(na), to_float(nb)
    if fa is not None and fb is not None:
        return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol
    return False

def build_prompt(tok, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve the following problem:\n\n{problem}"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    rows = apply_filters(load_rows(DATA_DIR)) ##### One line in hte test_data.json file

    per_level_correct, per_level_total = Counter(), Counter()
    correct = 0

    total_time = 0
    progress_bar = tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing")
    for i in progress_bar:
        start_time = time.time()
        batch = rows[i:i+BATCH_SIZE]
        prompts = [build_prompt(tok, r["problem"]) for r in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0),
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                top_p=TOP_P,
                eos_token_id=tok.eos_token_id,
            )
        dec = tok.batch_decode(gen, skip_special_tokens=False)
        for j, r in enumerate(batch):
            full = dec[j]
            resp = full[len(prompts[j]):] if full.startswith(prompts[j]) else full
            pred = extract_final_answer(resp)
            gold = extract_final_answer(r.get("solution", r.get("answer","")))  ##### r is a dictionary and ".get(key, default)" tries to return the value for key, or default if the key doesn’t exist.
            ok = approx_equal(pred, gold)
            lvl = r.get("level", "unknown")
            per_level_total[lvl] += 1
            if ok:
                per_level_correct[lvl] += 1
                correct += 1
        
        batch_time = time.time() - start_time
        total_time += batch_time
        progress_bar.set_postfix(Processed_Data=f"{min(i+len(batch), len(rows))}/{len(rows)}", Running_Acc=f"{(correct/(i+len(batch))*100):.2f}", Batch_Time=f"{batch_time:.2f}")

    overall = (correct / max(1, len(rows)))*100
    print("\n=== Final Results (vLLM) ===")
    print(f"Overall accuracy: {overall:.2f}  ({correct}/{len(rows)})")
    print(f"Total Time: {total_time/60.0:.3f} min")
    print("\nPer-level accuracy:")
    for lvl in sorted(per_level_total.keys()):
        acc = (per_level_correct[lvl] / per_level_total[lvl])*100
        print(f"  {lvl:>8}: {acc:.2f}  ({per_level_correct[lvl]}/{per_level_total[lvl]})")

if __name__ == "__main__":
    main()