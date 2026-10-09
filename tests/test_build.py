"""Sanity checks for the built datasets. usage: python tests/test_build.py  (set OLD_SPANS to compare paragraph indices)"""
import os, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_from_disk
import config, build_datasets as bd

cases = bd.load_cases(); bd.add_flags_and_split(cases); by_id = {c["case_id"]: c for c in cases}
D = {config.condition_id(c): load_from_disk(str(config.DS_DIR / config.condition_id(c))) for c in config.CONDITIONS}
refs = load_from_disk(str(config.DS_DIR / "_references"))

# 1) the first block of every article is its title
for lang in ("ru", "tr"):
    for rec in bd.read_jsonl(config.DATA_DIR / f"kuki_{lang}_aggregate.jsonl"):
        a, b = bd.paragraphs(rec["content"], lang)[0]
        assert rec["content"][a:b].strip() == rec["title"].strip(), rec["doc_id"]
print("ok  title block == title for all articles")

# 2) same cases, same order in every condition; no reference text leaks into prompts
ids = [r["case_id"] for r in next(iter(D.values()))]
assert all([r["case_id"] for r in d] == ids for d in D.values()); print("ok  identical case set and order in all", len(D), "conditions")
leaks = sum(len(c["insider"]) > 40 and any(c["insider"] in m["content"] for m in r["messages"])
            for r in next(iter(D.values())) for c in [by_id[r["case_id"]]])
assert leaks == 0; print("ok  no insider explanation (>40 chars) appears in the default prompts")

# 3) markers: exactly one pair, wrapping the phrase; none in unnamed prompts; none in the raw text
assert not any("[[" in c["content"] or "]]" in c["content"] for c in cases)
for cid, d in D.items():
    for r in d:
        user = r["messages"][1]["content"].replace("[[ ]]", "")   # drop the instruction line mentioning the markers
        if r["naming"] == "named" and r["context"] != "phrase":
            assert user.count("[[") == 1 and user.count("]]") == 1 and f"[[{r['phrase']}]]" in user, (cid, r["case_id"])
        else:
            assert "[[" not in user
print("ok  markers wrap the phrase exactly once (named, context>phrase); none elsewhere")

# 4) title handling and outlet swaps
d = D["ctx-paragraph__title-1__outlet-none__en__named"]
assert all(("Title: " in r["messages"][1]["content"]) == (not r["span_in_title"]) for r in d); print("ok  title line present unless span lies in the title")
for r in D["ctx-paragraph__title-0__outlet-swapped__en__named"]:
    assert r["pole_shown"] != r["pole_true"] and r["outlet_shown"] != r["source"]
assert all(r["outlet_shown"] == r["source"] for r in D["ctx-paragraph__title-0__outlet-true__en__named"]); print("ok  swapped outlet always from the opposite pole")
print("    swapped pairs (true -> shown):", dict(collections.Counter((r["source"], r["outlet_shown"]) for r in D["ctx-paragraph__title-0__outlet-swapped__en__named"])))

# 5) paragraph index agrees with the S1-S5 pipeline (optional)
old = os.environ.get("OLD_SPANS")
if old and Path(old).exists():
    import pandas as pd
    o = pd.read_csv(old); o["key"] = o.lang + ":" + o.article_id + ":" + o.annotator
    hits = tot = 0
    for c in cases:
        row = o[(o.key == f"{c['lang']}:{c['article_id']}:{c['annotator']}") & (o.start.sub(c["start"]).abs() < 3)]
        if len(row):
            paras = bd.paragraphs(c["content"], c["lang"]); first = bd.overlapping(paras, c["start"], c["end"])[0]
            tot += 1; hits += int(first == row.iloc[0].paragraph)
    print(f"ok  paragraph index matches old pipeline for {hits}/{tot} cases"); assert hits / tot > 0.98

# 6) counts
r = refs.to_pandas()
print("\ncases per lang:", r.lang.value_counts().to_dict(), "| split:", r.split.value_counts().to_dict())
print("dev per lang:", r[r.split == "dev"].lang.value_counts().to_dict())
print("multiref cases (phrase explained by >=2 annotators):", r[r.is_multiref].groupby("lang").size().to_dict(),
      "| distinct phrases:", r[r.is_multiref].assign(p=lambda x: x.phrase.str.casefold().str.strip()).groupby("lang").p.nunique().to_dict())
print("same-article multiref cases:", r[r.same_article_multiref].groupby("lang").size().to_dict())
print("flags: span_in_title", r.span_in_title.sum(), "| crosses_sentence", r.crosses_sentence.sum(), "| crosses_paragraph", r.crosses_paragraph.sum())
print("per annotator:", r.annotator.value_counts().to_dict())
