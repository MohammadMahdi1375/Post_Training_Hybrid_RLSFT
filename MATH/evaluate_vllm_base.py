import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

import os, re, json, glob, math, time
from typing import List, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---- Config via env ----
MODEL_PATH     = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B")
DATA_DIR       = os.environ.get("DATA_DIR", "/home/mohammad-m/TTT/Post-Training/MATH/data")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 64))     # prompts per vLLM generate() call
TEMPERATURE    = float(os.environ.get("TEMPERATURE", 0.0))
TOP_P          = float(os.environ.get("TOP_P", 1.0))
LIMIT          = os.environ.get("LIMIT")                   # e.g., "500"
FILTER_LEVELS  = os.environ.get("FILTER_LEVELS")           # e.g., "1,2,3"
ONLY_EASY      = os.environ.get("ONLY_EASY", "0") == "1"
ONLY_HARD      = os.environ.get("ONLY_HARD", "0") == "1"
TP_SIZE        = int(os.environ.get("TP_SIZE", 1))         # vLLM tensor parallel size
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", 4096))
DTYPE          = os.environ.get("DTYPE", "float16")       # "bfloat16" or "float16"
FEW_SHOT       = os.environ.get("FEW_SHOT", "1") == "1"


def read_json_or_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()          # read ALL characters
        # JSONL (one JSON object per line)
        if "\n" in content and content.strip().startswith("{") and "\n{" in content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        else:
            # Normal JSON (dict or list)
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
    s = s.strip()
    s = re.sub(r"\\boxed\s*\{([^{}]|{[^{}]*})*\}", lambda m: m.group(0)[7:-1], s)
    s = s.replace("$$", "$").strip("$ ").replace(" ", "").replace(",", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\(d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\2)/(\3)", s)
    if s.endswith("."): s = s[:-1]
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
    def to_float(x: str) -> Optional[float]:
        try:
            if "/" in x and not any(c.isalpha() for xch in x):
                p, q = x.split("/", 1); return float(p)/float(q)
            return float(x)
        except Exception: return None
    fa, fb = to_float(na), to_float(nb)
    if fa is not None and fb is not None:
        return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol
    return False

def build_prompt(tok, problem: str) -> str:
    if FEW_SHOT:
        demo = (
            "You are a math solver. Show steps and put the final answer in \\boxed{...}.\n\n"
            "Example:\n"
            "Problem:\nCompute: 3^4 - 5\\cdot 8.\n"
            "Solution:\n3^4 = 81 and 5\\cdot 8 = 40, so 81 - 40 = \\boxed{41}.\n\n"
        )
    else:
        demo = "You are a math solver. Show steps and put the final answer in \\boxed{...}.\n\n"

    return demo + "Problem:\n" + problem + "\n\nSolution:"

def main():
    # Tokenizer only for prompt rendering (no HF model here)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Initialize vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,  # "bfloat16" or "float16"
    )

    rows = apply_filters(load_rows(DATA_DIR))  # One line per sample in test_data.json

    per_level_correct, per_level_total = Counter(), Counter()
    correct = 0
    total_time = 0.0

    # vLLM will micro-batch internally; we still chunk the list to control memory
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=None,  # let EOS stop
    )

    progress_bar = tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing")
    for i in progress_bar:
        start_time = time.time()
        batch = rows[i:i+BATCH_SIZE]
        prompts = [build_prompt(tok, r["problem"]) for r in batch]

        # vLLM generate — returns a list of RequestOutput objects (same order as prompts)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)

        for r, out in zip(batch, outputs):
            text = out.outputs[0].text if out.outputs else ""
            pred = extract_final_answer(text)
            gold = extract_final_answer(r.get("solution", r.get("answer","")))
            ok = approx_equal(pred, gold)
            lvl = r.get("level", "unknown")
            per_level_total[lvl] += 1
            if ok:
                per_level_correct[lvl] += 1
                correct += 1

        batch_time = time.time() - start_time
        total_time += batch_time
        progress_bar.set_postfix(Processed_Data=f"{min(i+len(batch), len(rows))}/{len(rows)}", Running_Acc=f"{(correct/(i+len(batch))*100):.2f}", Batch_Time=f"{batch_time:.2f}")


    overall = correct / max(1, len(rows))
    print("\n=== Final Results (vLLM) ===")
    print(f"Overall accuracy: {overall:.2f}  ({correct*100}/{len(rows)})")
    print(f"Total Time: {total_time/60.0:.3f} min")
    print("\nPer-level accuracy:")
    for lvl in sorted(per_level_total.keys()):
        acc = per_level_correct[lvl] / per_level_total[lvl]
        print(f"  {lvl:>8}: {acc:.4f}  ({per_level_correct[lvl]}/{per_level_total[lvl]})")

if __name__ == "__main__":
    main()
