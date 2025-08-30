import json
from pathlib import Path
from typing import Optional

# -------- CONFIG --------
BASE_DIR = Path("/home/mohammad-m/TTT/Post-Training/MATH/data/train")  # set this to your MATH root
OUT_EASY = Path("train_data_easy.jsonl")
OUT_HARD = Path("train_data_hard.jsonl")
# Auto-discover category dirs (handles precalculus & any future additions)
CATEGORY_DIRS = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
# ------------------------

def extract_last_boxed_balanced(s: str) -> Optional[str]:
    if not s:
        return None
    target = r"\boxed{"
    last = s.rfind(target)
    if last == -1:
        return None
    i = last + len(target)
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
                break
            buf.append(c)
        else:
            buf.append(c)
        i += 1
    if depth != 0:
        return None
    ans = "".join(buf).strip()
    return ans.rstrip(" $.\n\r\t") or None

def parse_level_to_int(level_str: Optional[str]) -> Optional[int]:
    # Handles formats like "Level 1", "Level 3", "Level 2-3"
    if not level_str:
        return None
    num = ""
    for ch in level_str:
        if ch.isdigit():
            num += ch
            # take only the first contiguous number (e.g., "2-3" -> "2")
            # break if next char isn’t digit to avoid reading "23" in "2-3"
        elif num:
            break
    return int(num) if num else None

def load_one_json(path: Path):
    # Try regular JSON (dict or list)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except json.JSONDecodeError:
        # Fallback to JSONL
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
                    pass
        return rows

def main():
    idx = 0
    n_easy = 0
    n_hard = 0

    with OUT_EASY.open("w", encoding="utf-8") as feasy, OUT_HARD.open("w", encoding="utf-8") as fhard:
        for cat_dir in CATEGORY_DIRS:
            for fp in sorted(cat_dir.rglob("*.json")):
                for obj in load_one_json(fp):
                    problem = obj.get("problem")
                    level = obj.get("level")
                    typ = obj.get("type")
                    sol_raw = obj.get("solution", "")
                    final = extract_last_boxed_balanced(sol_raw)
                    level_num = parse_level_to_int(level)

                    rec = {
                        "idx": idx,
                        "problem": problem,
                        "level": level,
                        "type": typ,
                        "solution": final,          # ONLY the \boxed{...} content
                        "category": cat_dir.name,   # keep original category too (optional)
                    }

                    if level_num is not None and level_num < 4:
                        feasy.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_easy += 1
                    else:
                        fhard.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_hard += 1

                    idx += 1

    print(f"[done] total={idx}, easy(<4)={n_easy}, hard(>=4 or unknown)={n_hard}")
    print(f"easy -> {OUT_EASY.resolve()}")
    print(f"hard -> {OUT_HARD.resolve()}")

if __name__ == "__main__":
    main()
