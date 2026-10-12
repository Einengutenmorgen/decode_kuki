"""Hand-code the errors of ONE model on ONE case across the four context conditions (phrase, sentence, paragraph, article).
Each group shows the reference once and the four explanations together, in random order with the context hidden, so you can compare them directly.
  python label_ctx_errors.py build [--models gemma4-31b olmo3-7b-instruct] [--groups 15] [--judge-model openai/gpt-6-luna-20260922]
  python label_ctx_errors.py label [--show-context]       # interactive, resumable
  python label_ctx_errors.py summary
Per output type one or more codes, e.g.   g h      ok      = (same codes as the previous output)
  new code on the fly:   ## key description            e.g.  w ## m missed allusion: literary or pop-culture reference not recognised
  free comment:          # text                         e.g.  g h # reference looks thin
  same label for all outputs of the group left to code:   5 <codes>     e.g.  5 w h # same error in every context
Commands: skip (comes back later) | b undo the last label | r show the group again | ? codebook | q quit.
Groups: test split, one model, the four contexts judged; half have mixed verdicts (context changes the verdict), half are errors in every context.
Files in analysis/error_coding/: ctx_sample.csv (items), ctx_labels.csv (your labels, saved after every output), ctx_codes.json (codebook, grows with ##)."""
import argparse, json, random, re, textwrap
from pathlib import Path
import pandas as pd
import config
from judge import load_refs, ANCHOR_PHRASES
from analysis import DEFAULT, cid

DIR = config.ROOT / "analysis" / "error_coding"
SAMPLE, LABELS, CODEBOOK = DIR / "ctx_sample.csv", DIR / "ctx_labels.csv", DIR / "ctx_codes.json"
CONTEXTS = ["phrase", "sentence", "paragraph", "article"]
CIDS = {"phrase": cid(context="phrase"), "sentence": cid(context="sentence"), "paragraph": DEFAULT, "article": cid(context="article")}
RESERVED = {"skip", "b", "q", "r", "?", "="}
SEED_CODES = {"ok": "no error: the explanation matches the reference's core meaning",
              "x": "superset: contains the reference's core meaning and adds more that looks plausible or true (reference is thin)",
              "a": "alternative reading: plausible different insider reading (reference thin or ambiguous)",
              "g": "generic: right domain or direction, but misses the specific referent, saying or event",
              "w": "wrong referent: a different person, group, event or saying",
              "s": "no signal: names the referent but misses what it signals (stance, allusion, implication)",
              "m": "missed allusion: a literary, historical or pop-culture reference (character, saying, event) is not recognised",
              "h": "hallucinated detail: added names, dates or events that look invented or cannot be supported",
              "l": "literal: mostly restates the surface wording",
              "t": "mistranslation: wrong sense of a word or other wording error",
              "o": "other: say what in the note"}
COLS = ["gid", "slot", "context", "model", "case_id", "verdict", "codes", "note"]

def build(args):
    refs, rows = load_refs(), []
    for f in Path(args.judgments).glob("*__j3.jsonl"):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows); df = df[(df.judge_model == args.judge_model) & df.condition_id.isin(CIDS.values()) & df.model.isin(args.models) & (df.run == 0)]
    df = df.drop_duplicates(["condition_id", "model", "case_id"], keep="last")
    df = df[[refs[k]["split"] == "test" and refs[k]["phrase"] not in ANCHOR_PHRASES for k in df.case_id]]
    df = df.assign(context=df.condition_id.map({v: k for k, v in CIDS.items()}))
    rng, groups = random.Random(config.SEED), {}
    for (model, case), x in df.groupby(["model", "case_id"]):
        if len(x) < 4 or x.verdict.isin(["invalid", "judge_error"]).any(): continue      # need four judged outputs with text
        eq = (x.verdict == "equivalent").sum()
        if eq < 4: groups.setdefault(model, {"mixed": [], "all_err": []})["mixed" if eq else "all_err"].append(x)
    chosen = []
    for model in args.models:
        g = groups.get(model, {"mixed": [], "all_err": []}); k = args.groups // 2
        a, b = rng.sample(g["mixed"], min(k, len(g["mixed"]))), rng.sample(g["all_err"], min(args.groups - k, len(g["all_err"])))
        extra = [x for x in g["mixed"] + g["all_err"] if not any(x is y for y in a + b)]     # fill up if one pattern ran short
        fill_up = rng.sample(extra, min(max(args.groups - len(a) - len(b), 0), len(extra)))
        chosen += [(x, "mixed") for x in a] + [(x, "all_err") for x in b] + [(x, "fill") for x in fill_up]
        print(model, "groups:", len(a), "mixed +", len(b), "all-error +", len(fill_up), "fill")
    rng.shuffle(chosen); items = []
    for i, (x, pattern) in enumerate(chosen):
        c = refs[x.case_id.iloc[0]]
        for slot, ctx in enumerate(rng.sample(CONTEXTS, 4), start=1):                      # random order, context hidden while coding
            r = x[x.context == ctx].iloc[0]
            items.append(dict(gid=f"g{i:03d}", slot=slot, context=ctx, model=r.model, case_id=r.case_id, lang=c["lang"], phrase=c["phrase"],
                              literal=c["literal"], reference=c["insider"], ref_annotator=c["annotator"], candidate=r.candidate, verdict=r.verdict,
                              literal_overlap=r.literal_overlap, extra=bool(r.extra_content), missing=bool(r.missing_content), reason=r.reason, pattern=pattern))
    DIR.mkdir(parents=True, exist_ok=True); pd.DataFrame(items).to_csv(SAMPLE, index=False)
    print(f"wrote {len(items)} outputs in {len(chosen)} groups to {SAMPLE}")

def fill(label, text, width=100):
    return textwrap.fill(str(text), width, initial_indent=f"{label:<14}", subsequent_indent=" " * 14)

def show_group(rows, show_context):
    r0 = rows[0]; print("\n" + "=" * 100 + f"\n{r0['gid']} | {r0['lang']}")
    for k, v in (("PHRASE", r0["phrase"]), ("LITERAL", r0["literal"]), ("REFERENCE", r0["reference"])): print(fill(k, v))
    for r in rows:
        ctx = f" [{r['context']}]" if show_context else ""
        print("-" * 100 + f"\nOUTPUT {r['slot']}{ctx} | judge: {r['verdict']} | adds idea: {r['extra']} | lacks reference idea: {r['missing']} | round 2: {r['literal_overlap']}")
        print(fill("EXPLANATION", r["candidate"])); print(fill("JUDGE SAYS", r["reason"]))

def overview(codes):
    """Compact codebook plus input hints, printed above every prompt so nothing has to be scrolled back to."""
    short = " | ".join(f"{k} {v.split(':')[0][:36]}" for k, v in codes.items())
    return ("-" * 100 + "\n" + textwrap.fill(short, 100, initial_indent="codes: ", subsequent_indent="       ")
            + "\n5 <codes> = same for all outputs of this group | = same as previous | ## key description = new code | # note | skip  b  r  ?  q")

def parse(s, codes):
    """'g h ## key description # note' -> (code list, {new key: description}, note); raises ValueError with a message."""
    body, *rest = re.split(r"(?<!#)#(?!#)", s, maxsplit=1); note = rest[0].strip() if rest else ""
    known, *defs = body.split("##"); new = {}
    for d in defs:
        key, _, desc = d.strip().partition(" ")
        if not key or key in RESERVED: raise ValueError(f"cannot use {key!r} as a code name")
        if key in codes or key in new: raise ValueError(f"'{key}' already exists, just type it (? shows the codebook)")
        new[key] = desc.strip() or key
    chosen = known.split() + list(new)
    bad = [c for c in chosen if c not in codes and c not in new and c != "="]
    if bad or not chosen: raise ValueError(f"unknown or missing code {bad or ''}: use ? for the codebook, ## key description for a new one")
    if "ok" in chosen and len(chosen) > 1: raise ValueError("ok cannot be combined with other codes")
    return chosen, new, note

def label(args):
    sample = pd.read_csv(SAMPLE, keep_default_na=False)
    groups = {gid: x.sort_values("slot").to_dict("records") for gid, x in sample.groupby("gid", sort=False)}
    codes = json.load(open(CODEBOOK, encoding="utf-8")) if CODEBOOK.exists() else dict(SEED_CODES)
    labels = pd.read_csv(LABELS, keep_default_na=False).to_dict("records") if LABELS.exists() else []
    save = lambda: pd.DataFrame(labels, columns=COLS).to_csv(LABELS, index=False)
    book = lambda: "\n".join(f"  {k:<4}{v}" for k, v in codes.items())
    done = {(l["gid"], int(l["slot"])) for l in labels}
    todo = [(g, r["slot"]) for g, rows in groups.items() for r in rows if (g, r["slot"]) not in done]
    print("codebook:\n" + book() + f"\n{len(todo)} outputs left in {len({g for g, _ in todo})} groups of {len(groups)}")
    shown, batches = None, []                                              # batches: how many outputs each accepted entry labelled (for undo)
    while todo:
        gid, slot = todo[0]; r = groups[gid][slot - 1]
        if shown != gid: show_group(groups[gid], args.show_context); shown = gid
        print(overview(codes))
        s = input(f"[{gid} output {slot}/4] code(s) > ").strip()
        if s == "q": break
        if s == "?": print(book()); continue
        if s == "r": shown = None; continue
        if s == "skip": todo.append(todo.pop(0)); shown = None; continue
        if s == "b":
            back = [labels.pop() for _ in range(min(batches.pop() if batches else 1, len(labels)))]
            todo[:0] = [(l["gid"], int(l["slot"])) for l in reversed(back)]; save(); shown = None
            continue
        everything = s.split()[:1] == ["5"]                                   # `5 <codes>`: one label for all outputs of this group still to code
        if everything: s = s.partition("5")[2].strip()
        try: chosen, new, note = parse(s, codes)
        except ValueError as e: print(e); continue
        if "=" in chosen:                                                     # same codes as the previous output of this group
            prev = [l for l in labels if l["gid"] == gid]
            if not prev or len(chosen) > 1: print("= must stand alone and needs a previous output in this group"); continue
            chosen = prev[-1]["codes"].split()
        if new: codes.update(new); json.dump(codes, open(CODEBOOK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        targets = [t for t in todo if t[0] == gid] if everything else [(gid, slot)]
        for g, sl in targets:
            x = groups[g][sl - 1]
            labels.append(dict(gid=g, slot=sl, context=x["context"], model=x["model"], case_id=x["case_id"], verdict=x["verdict"], codes=" ".join(chosen), note=note))
        todo[:] = [t for t in todo if t not in targets]; batches.append(len(targets)); save()
    print(f"{len(labels)} outputs labelled, {len(todo)} left")

def summary(args):
    l = pd.read_csv(LABELS, keep_default_na=False)
    e = l.assign(code=l.codes.str.split()).explode("code").reset_index(drop=True)
    n = l.context.value_counts().reindex(CONTEXTS, fill_value=0)
    print(f"outputs: {len(l)} in {l.gid.nunique()} groups | per context: {n.to_dict()}")
    print("\ncode counts per context (an output can carry several codes):\n", pd.crosstab(e.code, e.context).reindex(columns=CONTEXTS, fill_value=0))
    print("\nshare of outputs per context:\n", (pd.crosstab(e.code, e.context).reindex(columns=CONTEXTS, fill_value=0) / n.replace(0, 1)).round(2))
    print("\njudge vs you (rows: judge verdict, columns: your coding):\n", pd.crosstab(l.verdict, l.codes.eq("ok").map({True: "ok", False: "error"})))
    codes = {gid: {r.context: frozenset(r.codes.split()) for r in x.itertuples()} for gid, x in l.groupby("gid")}
    full = {g: c for g, c in codes.items() if set(c) == set(CONTEXTS)}
    ok = frozenset({"ok"})
    print(f"\ngroups fully coded: {len(full)}"
          f" | same error codes in all four contexts: {sum(len(set(c.values())) == 1 and ok not in c.values() for c in full.values())}"
          f" | phrase has an error but paragraph is ok: {sum(c['phrase'] != ok and c['paragraph'] == ok for c in full.values())}"
          f" | paragraph has an error but phrase is ok: {sum(c['paragraph'] != ok and c['phrase'] == ok for c in full.values())}")
    print("\nper group (codes by context):")
    for gid, x in l.groupby("gid", sort=False): print(f"  {gid} {x.model.iloc[0]:<18}" + " | ".join(f"{r.context}: {r.codes}" for r in x.sort_values('context', key=lambda s: s.map(CONTEXTS.index)).itertuples()))

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--models", nargs="+", default=["gemma4-31b", "olmo3-7b-instruct"]); b.add_argument("--groups", type=int, default=15)
    b.add_argument("--judgments", default=str(config.JUDGE_DIR)); b.add_argument("--judge-model", default="openai/gpt-6-luna-20260922"); b.set_defaults(f=build)
    l = sub.add_parser("label"); l.add_argument("--show-context", action="store_true"); l.set_defaults(f=label)
    sub.add_parser("summary").set_defaults(f=summary)
    args = ap.parse_args(); args.f(args)

if __name__ == "__main__":
    main()