"""Hand-code why model explanations miss the annotator's insider meaning (test split, default condition, verdict not equivalent),
plus a few annotator-vs-annotator disagreements coded with the same codes for comparison.
  python label_errors.py build [--per-model 12] [--annotator 15] [--judge-model openai/gpt-6-luna-20260922]   # fixed, seeded sample
  python label_errors.py label                                                                                  # interactive, resumable
  python label_errors.py summary                                                                                # code counts per source
Per item type one or two codes, e.g.  `g h`  or  `a # reference is very thin`  (text after # is a free note).
Commands: s = skip (comes back later), b = undo the last label, ? = codebook, q = quit. The source (which model, or annotator) is hidden while you code.
Files: analysis/error_coding/error_sample.csv (items) and error_labels.csv (your labels, saved after every item)."""
import argparse, json, random, textwrap
from pathlib import Path
import pandas as pd
import config
from judge import load_refs, ANCHOR_PHRASES
from analysis import DEFAULT

DIR = config.ROOT / "analysis" / "error_coding"
CODES = {"e": "judge too strict: the candidate is actually equivalent, no real error",
         "a": "alternative reading: plausible different insider reading (reference thin or ambiguous)",
         "g": "generic: right domain or direction, but misses the specific referent, saying or event",
         "w": "wrong referent: a different person, group, event or saying",
         "s": "no signal: names the referent but misses what it signals (stance, allusion, implication)",
         "h": "hallucinated or unsupported detail: invented-looking names, dates, events",
         "l": "literal: mostly restates the surface wording",
         "t": "mistranslation or wording error (wrong sense of a word)",
         "o": "other (say what in the note)"}
COLS = ["id", "model", "case_id", "run", "verdict", "codes", "note"]

def build(args):
    refs, rows = load_refs(), []
    for f in list(Path(args.judgments).glob("*__j3.jsonl")) + list(Path(args.judgments, "annotator").glob("*__j3.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows); df = df[df.judge_model == args.judge_model]
    df = df.drop_duplicates(["condition_id", "model", "case_id", "run"], keep="last")
    df = df[df.condition_id.isin([DEFAULT, "control-annotator"]) & df.verdict.isin(["partially_equivalent", "not_equivalent"])]
    df = df[[refs[k]["split"] == "test" and refs[k]["phrase"] not in ANCHOR_PHRASES for k in df.case_id]]
    seed, pick = config.SEED, []
    for model, x in df[df.model != "control"].groupby("model"):      # per model: a third not_equivalent, the rest partially_equivalent
        ne, pa = x[x.verdict == "not_equivalent"], x[x.verdict == "partially_equivalent"]
        k = min(len(ne), args.per_model // 3)
        pick += [ne.sample(k, random_state=seed), pa.sample(min(len(pa), args.per_model - k), random_state=seed)]
    a = df[df.model == "control"]; pick.append(a.sample(min(len(a), args.annotator), random_state=seed))
    pick = pd.concat(pick).sample(frac=1, random_state=seed)         # interleave sources
    items = []
    for i, r in enumerate(pick.itertuples()):
        c, is_ann = refs[r.case_id], r.model == "control"
        items.append(dict(id=f"e{i:03d}", model="annotator" if is_ann else r.model, case_id=r.case_id, run=r.run, lang=c["lang"],
                          phrase=c["phrase"], literal=c["literal"], reference=c["insider"], ref_annotator=c["annotator"],
                          candidate=r.candidate, verdict=r.verdict, literal_overlap=r.literal_overlap,
                          extra=bool(r.extra_content), missing=bool(r.missing_content), reason=r.reason))
    DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(DIR / "error_sample.csv", index=False)
    print(f"wrote {len(items)} items to {DIR / 'error_sample.csv'}:", pd.Series([i["model"] for i in items]).value_counts().to_dict())

def show(r, pos, total):
    print("\n" + "=" * 100)
    print(f"[{pos}/{total}] {r['id']} | {r['lang']} | judge: {r['verdict']} | candidate adds idea: {r['extra']} | lacks reference idea: {r['missing']} | round 2: {r['literal_overlap']}")
    for k, v in (("PHRASE", r["phrase"]), ("LITERAL", r["literal"]), ("REFERENCE", r["reference"]), ("CANDIDATE", r["candidate"]), ("JUDGE SAYS", r["reason"])):
        print(textwrap.fill(str(v), 100, initial_indent=f"{k:<12}", subsequent_indent=" " * 12))

def label(args):
    sample = pd.read_csv(DIR / "error_sample.csv", keep_default_na=False).to_dict("records"); by_id = {r["id"]: r for r in sample}
    out = DIR / "error_labels.csv"
    labels = pd.read_csv(out, keep_default_na=False).to_dict("records") if out.exists() else []
    save = lambda: pd.DataFrame(labels, columns=COLS).to_csv(out, index=False)
    queue = [r for r in sample if r["id"] not in {l["id"] for l in labels}]
    print("codes:\n" + "\n".join(f"  {k}  {v}" for k, v in CODES.items()) + f"\n{len(queue)} items left of {len(sample)}")
    while queue:
        r = queue[0]; show(r, len(labels) + 1, len(sample))
        s = input("code(s) [? s b q] > ").strip()
        if s == "q": break
        if s == "?": print("\n".join(f"  {k}  {v}" for k, v in CODES.items())); continue
        if s == "s": queue.append(queue.pop(0)); continue
        if s == "b":
            if labels: queue.insert(0, by_id[labels.pop()["id"]]); save()
            continue
        head, _, note = s.partition("#"); codes = head.split()
        if not codes or any(c not in CODES for c in codes): print("need one or two known codes (? shows them)"); continue
        labels.append(dict(id=r["id"], model=r["model"], case_id=r["case_id"], run=r["run"], verdict=r["verdict"], codes=" ".join(codes), note=note.strip()))
        save(); queue.pop(0)
    print(f"{len(labels)} labelled, {len(queue)} left")

def summary(args):
    l = pd.read_csv(DIR / "error_labels.csv", keep_default_na=False)
    e = l.assign(code=l.codes.str.split()).explode("code").reset_index(drop=True)
    n = l.model.value_counts()
    print(f"items: {len(l)} | judge too strict (code e): {(e.code == 'e').sum()} of {len(l)}")
    print("\ncounts (an item can carry two codes):\n", pd.crosstab(e.code, e.model, margins=True))
    print("\nshare of items per source:\n", (pd.crosstab(e.code, e.model) / n).round(2))
    print("\nitems per source:", n.to_dict())

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--per-model", type=int, default=12); b.add_argument("--annotator", type=int, default=15)
    b.add_argument("--judgments", default=str(config.JUDGE_DIR)); b.add_argument("--judge-model", default="openai/gpt-6-luna-20260922"); b.set_defaults(f=build)
    sub.add_parser("label").set_defaults(f=label); sub.add_parser("summary").set_defaults(f=summary)
    args = ap.parse_args(); args.f(args)

if __name__ == "__main__":
    main()
