"""Olmo-3's chat template calls `tools | tojson` when `tools` is undefined, which crashes vLLM 0.31's startup probe.
This writes a copy that treats an undefined `tools` like None and checks that the rendered prompts are identical
to the original template for every prompt in our datasets (with `tools` omitted and with tools=None).
usage (repo root): python tools/make_olmo_template.py [hf_id [output.jinja]]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_from_disk
from transformers import AutoTokenizer
import config

MODEL = sys.argv[1] if len(sys.argv) > 1 else "allenai/Olmo-3-7B-Instruct"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("olmo3_template_patched.jinja")
OUT.parent.mkdir(parents=True, exist_ok=True)
orig_tok, patched_tok = AutoTokenizer.from_pretrained(MODEL), AutoTokenizer.from_pretrained(MODEL)
old = orig_tok.chat_template
new = old.replace("tools is not none", "(tools is defined and tools is not none)").replace("tools is none", "(tools is not defined or tools is none)")
assert new != old, "pattern not found: the template changed, inspect it first"
patched_tok.chat_template = new; OUT.write_text(new, encoding="utf-8")

n = bad = 0
for d in sorted(config.DS_DIR.iterdir()):
    if d.name.startswith("_"): continue
    for r in load_from_disk(str(d)):
        ref = orig_tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        for kw in ({}, {"tools": None}):
            n += 1; bad += ref != patched_tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **kw)
print(f"wrote {OUT}; {n - bad}/{n} renders identical to the original template")
sys.exit(1 if bad else 0)