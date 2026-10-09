"""Offline checks for judge.py (no judge model needed). usage (repo root): python tests/test_judge.py"""
import collections, hashlib, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config, judge

refs = judge.load_refs()
# 1) repeat runs: first occurrence is run 0, extra copies only with --runs 2
rows = [{"case_id": "a", "raw": "1"}, {"case_id": "b", "raw": "2"}, {"case_id": "a", "raw": "3"}]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f: f.write("\n".join(map(json.dumps, rows)))
assert [(r["case_id"], r["run"]) for r in judge.load_results(f.name, 1)] == [("a", 0), ("b", 0)]
assert len(judge.load_results(f.name, 2)) == 3; print("ok  first occurrence kept, repeats only with --runs 2")

# 2) labels and schema
assert judge.VERDICTS == ["not_equivalent", "partially_equivalent", "equivalent"]
assert judge.SCHEMA["properties"]["verdict"]["enum"] == judge.VERDICTS and all(v in judge.RUBRIC for v in judge.VERDICTS)
assert "reference" not in judge.RUBRIC.lower() and "candidate" not in judge.RUBRIC.lower(); print("ok  three renamed labels; the prompt never says which explanation is the reference")

# 3) symmetric prompt: A/B order is deterministic, ~50/50, and the same explanations appear in either order
c = next(iter(refs.values()))
m1, m2 = judge.user_message(c, "CAND", True), judge.user_message(c, "CAND", False)
assert f"Explanation A: {c['insider']}\nExplanation B: CAND" in m1 and f"Explanation A: CAND\nExplanation B: {c['insider']}" in m2
order = judge.order_map(refs); assert order == judge.order_map(refs)
for split in ("dev", "test"):
    for lang in ("ru", "tr"):
        v = [order[k] for k, r in refs.items() if r["split"] == split and r["lang"] == lang]
        assert abs(sum(v) - (len(v) - sum(v))) <= 1, (split, lang)
print("ok  A/B order deterministic and balanced (+-1) within dev/test x ru/tr")

# 4) controls unchanged: same fingerprints as before the prompt change
expected = {("literal", "dev"): "ef9cc967e40dd1fa", ("literal", "test"): "15048a5b4bc6518b", ("literal", "all"): "0b3f5d824fa4ca6f",
            ("shuffled", "dev"): "8e8e4297c38b2900", ("shuffled", "test"): "3aa0c331021c3ddb", ("shuffled", "all"): "81c21bc58412cfce"}
for (kind, split), h in expected.items():
    t = judge.control_tasks(kind, refs, split)
    assert hashlib.sha256(json.dumps(t, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16] == h, (kind, split)
print("ok  control tasks identical to the pre-change fingerprints (6/6)")

# 5) parse failures -> invalid without a judge call; unnamed conditions are skipped by `run`
bad = judge.tasks_from_results([{"case_id": next(iter(refs)), "condition_id": "x__named", "model": "m", "run": 0, "raw": "oops", "parse_ok": False}], refs, "all")
assert bad[0]["verdict"] == "invalid"; print("ok  parse failures -> invalid")
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as g:
    g.write(json.dumps({"case_id": next(iter(refs)), "condition_id": "ctx-paragraph__title-0__outlet-none__en__unnamed", "model": "m", "raw": "{}", "parse_ok": True}))
import argparse, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): judge.cmd_run(argparse.Namespace(results=g.name, runs=1, split="all", dry_run=True, judge_model=None, workers=1, base_url=None, no_schema=False))
assert "skipping" in buf.getvalue() and not (config.JUDGE_DIR / (Path(g.name).stem + "__dry-run__j2.jsonl")).exists(); print("ok  unnamed condition skipped, no judgment file written")

# 6) kappa
assert abs(judge.kappa([0, 1, 2, 2], [0, 1, 2, 2]) - 1) < 1e-9
assert abs(judge.kappa([0, 0, 1, 1], [0, 1, 1, 1], k=2) - 0.5) < 1e-9; print("ok  kappa")

# 7) rubric anchors exist and are excluded from validation
dev = {c["phrase"] for c in refs.values() if c["split"] == "dev"}
assert judge.ANCHOR_PHRASES <= dev; print("ok  the 3 anchor phrases are dev cases (excluded from the validation sample)")
