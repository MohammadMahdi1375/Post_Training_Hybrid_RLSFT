import json
from pathlib import Path
import sys
from typing import Optional

# -------- CONFIG --------
BASE_DIR = Path("/home/mohammad-m/TTT/Post-Training/MATH/data/train")  # <- set this to your MATH root
CATEGORIES = [
    "prealgebra",
    "algebra",
    "number_theory",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "precalculus",
]
OUTFILE = Path("train_data.json")
# ------------------------

def extract_last_boxed_balanced(s: str) -> Optional[str]:
    """
    Return the content inside the *last* \boxed{...} in s.
    Uses balanced-brace scanning to support nested braces (e.g., \frac{...}).
    """
    if not s:
        return None
    target = r"\boxed{"
    last = s.rfind(target)
    if last == -1:
        return None
    i = last + len(target)  # position at first char after '{'
    depth = 1
    buf = []
    while i < len(s):
        c = s[i]
        if c == '{':
            depth += 1
            buf.append(c)
        elif c == '}':
            depth -= 1
            if depth == 0:
                # reached matching close for the \boxed{ ... }
                break
            buf.append(c)
        else:
            buf.append(c)
        i += 1
    if depth != 0:
        # unmatched braces; give up
        return None
    ans = "".join(buf).strip()
    # strip very common trailing tokens people sometimes add after math mode
    ans = ans.rstrip(" $.\n\r\t")
    return ans or None

def load_one_json(path: Path):
    """
    Return a list of dicts from a file:
      - if it's a single JSON object -> [obj]
      - if it's a list -> that list (only dicts are kept)
      - if it's JSONL -> list of dicts (one per line)
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except json.JSONDecodeError:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except json.JSONDecodeError:
                    # skip bad lines
                    pass
        return rows

def main():
    idx = 0
    # ensure deterministic order (sorted by category then filepath)
    files = []
    for cat in CATEGORIES:
        cat_dir = BASE_DIR / cat
        if not cat_dir.exists():
            print(f"[warn] missing category: {cat_dir}", file=sys.stderr)
            continue
        files.extend(sorted(cat_dir.rglob("*.json")))

    # stream-write JSONL
    with OUTFILE.open("w", encoding="utf-8") as out:
        for fp in files:
            for obj in load_one_json(fp):
                problem = obj.get("problem")
                level = obj.get("level")
                typ = obj.get("type")
                sol_raw = obj.get("solution", "")

                final = extract_last_boxed_balanced(sol_raw)

                rec = {
                    "idx": idx,
                    "problem": problem,
                    "level": level,
                    "type": typ,
                    "solution":sol_raw,
                    "answer": final,  # ONLY the content inside \boxed{...}
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx += 1

    print(f"[done] wrote {idx} records to {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()
