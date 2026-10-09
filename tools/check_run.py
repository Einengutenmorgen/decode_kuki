"""Check (and clean) the generation results of one model x condition x split.
exit 0 = complete and clean | 1 = cases missing or API errors (a re-run helps) | 2 = too many unparsable outputs (a re-run will not help)
usage: python tools/check_run.py --model <alias> --condition <id> --split dev|test|all [--purge-errors] [--max-parse-fail 0.10] [--quiet]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_from_disk
import config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--condition", required=True)
ap.add_argument("--split", required=True, choices=["dev", "test", "all"]); ap.add_argument("--purge-errors", action="store_true")
ap.add_argument("--max-parse-fail", type=float, default=0.10); ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

ds = load_from_disk(str(config.DS_DIR / a.condition)).select_columns(["case_id", "split"])
expected = {r["case_id"] for r in ds if a.split == "all" or r["split"] == a.split}
path = config.RES_DIR / f"{a.condition}__{a.model}__{config.PROMPT_VERSION}.jsonl"
rows = []
if path.exists():
    for line in open(path, encoding="utf-8"):
        try: rows.append(json.loads(line))
        except Exception: pass                                   # half-written last line of an interrupted run
is_err = lambda r: str(r.get("raw", "")).startswith("ERROR")
n_err = sum(is_err(r) for r in rows if r["case_id"] in expected)
if a.purge_errors and n_err:                                     # the runner treats error rows as done, so remove them for a retry
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if not is_err(r): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [r for r in rows if not is_err(r)]
first = {}
for r in rows:
    if r["case_id"] in expected and not is_err(r): first.setdefault(r["case_id"], r)   # first copy of each case
missing, bad = len(expected) - len(first), sum(not r["parse_ok"] for r in first.values())
rate = bad / max(1, len(first))
code = 1 if (missing or n_err) else 2 if rate > a.max_parse_fail else 0
if not a.quiet or code:
    print(f"{a.model} | {a.condition} | {a.split}: {len(first)}/{len(expected)} cases, api errors {n_err}{' (purged)' if a.purge_errors and n_err else ''}, "
          f"unparsable {bad} ({rate:.1%}) -> {['ok', 'INCOMPLETE', 'TOO MANY UNPARSABLE'][code]}")
sys.exit(code)
