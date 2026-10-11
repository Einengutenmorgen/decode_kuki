"""Analysis of the L4 decoding experiment: judge verdicts -> scores per model x condition, paired condition effects, checks.
  python analysis.py [--split dev|test|all] [--judgments judgments] [--judge-version j3] [--judge-model <m>] [--out analysis/<split>]
Reads judgments/*__<judge-version>.jsonl (named conditions + the three controls), no model or API calls.
Score per judgment: not_equivalent 0 | partially_equivalent 1 | equivalent 2 | invalid (unparseable model output) 0; judge_error rows are dropped.
j3 adds, per judgment: equiv (strict: verdict equivalent), lit_equiv (round 2: candidate only restates the annotator's literal reading),
lit_beyond (round 2: clearly beyond the literal reading), extra / missing (candidate adds / lacks a distinct idea vs the annotator text).
All rates are over all cases incl. invalid outputs. `partially_equivalent` alone cannot be told from a literal translation (see the
literal control): rely on equiv, lit_beyond and the strata.
Strata (strata.csv, and the `stratum` column of contrasts.csv): from the reference control (annotator insider text as candidate), round 2:
`informative` = the annotator's insider text clearly goes beyond their literal reading; `low_info` = it mostly restates it; `unknown` = no control row.
The strata are judge-derived and come from the references only, never from model outputs.
Cases whose phrase is used as a rubric example (judge.ANCHOR_PHRASES) are excluded everywhere.
Repeat runs (judge.py --runs 2) are averaged per case. Uncertainty: bootstrap over phrases (cases sharing a phrase move together).
Contrasts are paired on case_id (same cases in both conditions) and always `other - base`.
Writes overview.csv, strata.csv, contrasts.csv, subsets.csv, checks.csv to --out and prints the same tables.
Human ceiling (annotator vs annotator on the variance set) is not part of this script: judge.py does not compare annotator pairs."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import config
from judge import JUDGE_VERSION, VERDICTS, ANCHOR_PHRASES

SCORE = {"not_equivalent": 0, "partially_equivalent": 1, "equivalent": 2, "invalid": 0}
META = ["lang", "cluster", "split", "is_multiref", "span_in_title", "pole_true", "annotator", "article_id"]
def cid(**kw): return config.condition_id(config.cond(**kw))
DEFAULT = cid()
CONTRASTS = [   # (axis, label, base, other)
    ("context", "phrase vs paragraph", DEFAULT, cid(context="phrase")),
    ("context", "sentence vs paragraph", DEFAULT, cid(context="sentence")),
    ("context", "article vs paragraph", DEFAULT, cid(context="article")),
    ("title", "title vs no title (paragraph)", DEFAULT, cid(title=True)),
    ("title", "title vs no title (phrase)", cid(context="phrase"), cid(context="phrase", title=True)),
    ("outlet", "true outlet vs none", DEFAULT, cid(outlet="true")),
    ("outlet", "swapped outlet vs none", DEFAULT, cid(outlet="swapped")),
    ("outlet", "swapped vs true outlet", cid(outlet="true"), cid(outlet="swapped")),
]

def load_judgments(args):
    rows = []
    for f in sorted(Path(args.judgments).glob(f"*__{args.judge_version}.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if not rows: raise SystemExit(f"no *__{args.judge_version}.jsonl in {args.judgments}")
    df = pd.DataFrame(rows)
    models = sorted(df.judge_model.unique())
    if args.judge_model: df = df[df.judge_model == args.judge_model]
    elif len(models) > 1: raise SystemExit(f"several judge models {models}; pick one with --judge-model")
    if df.empty: raise SystemExit(f"no judgments by {args.judge_model}")
    df = df.drop_duplicates(["condition_id", "model", "case_id", "run", "judge_model"], keep="last")   # retries are appended
    if (df.verdict == "judge_error").any(): print(f"warning: dropping {(df.verdict == 'judge_error').sum()} judge_error rows (rerun judge.py to fill them)")
    df = df[(df.verdict != "judge_error") & ~df.condition_id.str.endswith("__unnamed")]
    for k in ("literal_overlap", "extra_content", "missing_content"):
        if k not in df: df[k] = None                            # j2 files have no round-2 fields
    return df.assign(kind=np.where(df.model == "control", "control", "model"))

def reference_strata(df):
    """case_id -> informative | low_info | unknown, from round 2 of the reference control (does the annotator's insider text
    only restate their own literal reading?). Needs `judge.py control --kind reference` for the split."""
    r = df[df.condition_id == "control-reference"].drop_duplicates("case_id", keep="last")
    if r.empty: print("warning: no control-reference judgments, all cases get stratum 'unknown'")
    return {k: "unknown" if v is None or v != v else "informative" if v == "not_equivalent" else "low_info"
            for k, v in zip(r.case_id, r.literal_overlap)}

def load_meta():
    """Case metadata from the default condition dataset; pole_shown from the swapped-outlet dataset."""
    from datasets import load_from_disk
    m = load_from_disk(str(config.DS_DIR / DEFAULT)).to_pandas()[
        ["case_id", "lang", "article_id", "source", "pole_true", "annotator", "split", "is_multiref", "span_in_title", "phrase"]]
    m["cluster"] = m.lang + ":" + m.phrase.str.casefold().str.split().str.join(" ")
    sw = config.DS_DIR / cid(outlet="swapped")
    if sw.exists(): m = m.merge(load_from_disk(str(sw)).to_pandas()[["case_id", "pole_shown"]], on="case_id")
    return m

RATES = ["equiv", "partial_up", "lit_equiv", "lit_beyond", "extra", "missing", "invalid"]

def per_case(df, meta, split, strata):
    """One row per (model, condition, case): scores averaged over repeat runs."""
    df = df.assign(score=df.verdict.map(SCORE), equiv=df.verdict == "equivalent",
                   partial_up=df.verdict.isin(["equivalent", "partially_equivalent"]), invalid=df.verdict == "invalid",
                   lit_equiv=df.literal_overlap.eq("equivalent"), lit_beyond=df.literal_overlap.eq("not_equivalent"),
                   extra=df.extra_content.eq(True), missing=df.missing_content.eq(True))
    c = df.groupby(["kind", "model", "condition_id", "case_id"], as_index=False).agg(score=("score", "mean"), **{k: (k, "mean") for k in RATES})
    c = c.merge(meta, on="case_id", how="left")
    if c.lang.isna().any(): raise SystemExit("judgments contain case_ids missing from the datasets; rebuild with build_datasets.py?")
    n = len(c); c = c[~c.phrase.isin(ANCHOR_PHRASES)]
    if len(c) < n: print(f"note: excluded {n - len(c)} rows whose phrase is a rubric example")
    c = c.assign(stratum=c.case_id.map(strata).fillna("unknown"))
    return c if split == "all" else c[c.split == split]

def boot_ci(vals, clusters, B, rng):
    """95% cluster-bootstrap CI of the mean (ratio of cluster sums, so unequal cluster sizes are fine)."""
    codes, _ = pd.factorize(clusters); K = codes.max() + 1
    s, n = np.bincount(codes, np.asarray(vals, float), K), np.bincount(codes, minlength=K)
    idx = rng.integers(0, K, (B, K))
    return np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])

def with_all_langs(c): return pd.concat([c, c.assign(lang="all")], ignore_index=True)
def keyed(key, by): return dict(zip(by, key if isinstance(key, tuple) else (key,)))

def overview(c, B, rng):
    by, rows = ["kind", "model", "condition_id", "lang"], []
    for key, x in with_all_langs(c).groupby(by):
        lo, hi = boot_ci(x.score, x.cluster, B, rng)
        rows.append(dict(**keyed(key, by), n=len(x), **{k: x[k].mean() for k in RATES}, score=x.score.mean(), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)

def strata_table(c, B, rng):
    """Scores and rates by reference stratum (all languages pooled): the V2 claim rests on the `informative` rows."""
    by, rows = ["kind", "model", "condition_id", "stratum"], []
    for key, x in c.groupby(by):
        lo, hi = boot_ci(x.score, x.cluster, B, rng)
        rows.append(dict(**keyed(key, by), n=len(x), **{k: x[k].mean() for k in RATES}, score=x.score.mean(), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)

def contrasts(c, B, rng):
    out = []
    cols = ["model", "case_id", "stratum"] + META + (["pole_shown"] if "pole_shown" in c else [])
    for axis, label, base, other in CONTRASTS:
        a, b = (c[(c.kind == "model") & (c.condition_id == k)][cols + ["score"]] for k in (base, other))
        m = a.merge(b, on=cols, suffixes=("_base", "_other"))
        if axis == "title": m = m[~m.span_in_title]             # title spans: the title is the context, prompts are identical
        if m.empty: continue
        m = m.assign(diff=m.score_other - m.score_base)
        groupings = [["model", "lang"], ["model", "lang", "stratum"]] + ([["model", "lang", "pole_true"]] if axis == "outlet" else [])
        for by in groupings:
            for key, x in with_all_langs(m).groupby(by):
                lo, hi = boot_ci(x["diff"], x.cluster, B, rng)
                out.append(dict(axis=axis, contrast=label, **{"pole_true": "all", "stratum": "all", **keyed(key, by)}, n=len(x),
                                base=x.score_base.mean(), other=x.score_other.mean(), diff=x["diff"].mean(), ci_lo=lo, ci_hi=hi,
                                better=(x["diff"] > 0).sum(), worse=(x["diff"] < 0).sum(), same=(x["diff"] == 0).sum()))
    return pd.DataFrame(out)

def subsets(c, B, rng):
    """Default condition only: where do the scores differ (annotator, variance set, title spans)?"""
    d, rows = c[(c.kind == "model") & (c.condition_id == DEFAULT)], []
    for col in ("annotator", "is_multiref", "span_in_title"):
        for key, x in d.groupby(["model", "lang", col]):
            lo, hi = boot_ci(x.score, x.cluster, B, rng)
            rows.append(dict(subset=col, model=key[0], lang=key[1], value=key[2], n=len(x), score=x.score.mean(), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)

def checks(df, c):
    """Judge sanity: A/B position bias, run-to-run stability, and how far the controls sit from the models."""
    rows, d = [], df.assign(score=df.verdict.map(SCORE))
    for (kind, a_is), x in d.groupby(["kind", "a_is"]):
        rows.append(dict(check="position bias (mean score by who is Explanation A)", group=f"{kind}, A = {a_is}", n=len(x), value=x.score.mean()))
    if df.run.max() > 0:
        w = d.pivot_table(index=["model", "condition_id", "case_id"], columns="run", values="verdict", aggfunc="first").dropna(subset=[0, 1])
        for (model, _), x in w.groupby(level=[0, 1]):
            rows.append(dict(check="run-to-run: same verdict in run 0 and 1", group=model, n=len(x), value=(x[0] == x[1]).mean()))
    for kind, x in c[c.kind == "control"].groupby("condition_id"):
        rows.append(dict(check="control score (literal = floor, shuffled = chance, reference = 2.0)", group=kind, n=len(x), value=x.score.mean()))
        for k, what in (("equiv", "strict equivalent"), ("lit_equiv", "round 2 = only restates literal (literal ~1, shuffled ~0)"),
                        ("extra", "extra_content flag"), ("missing", "missing_content flag")):
            rows.append(dict(check=f"control: share with {what}", group=kind, n=len(x), value=x[k].mean()))
    m = c[(c.kind == "model") & (c.condition_id == DEFAULT)]
    for model, x in m.groupby("model"):
        rows.append(dict(check="default-condition score (compare with controls)", group=model, n=len(x), value=x.score.mean()))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--judgments", default=str(config.JUDGE_DIR)); ap.add_argument("--judge-version", default=JUDGE_VERSION)
    ap.add_argument("--judge-model"); ap.add_argument("--out"); ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(config.SEED)
    df = load_judgments(args); c = per_case(df, load_meta(), args.split, reference_strata(df))
    if c.empty: raise SystemExit(f"no judged cases in split {args.split}")
    df = df[df.case_id.isin(c.case_id)]
    out = Path(args.out or config.ROOT / "analysis" / args.split); out.mkdir(parents=True, exist_ok=True)
    tables = {"overview": overview(c, args.boot, rng), "strata": strata_table(c, args.boot, rng), "contrasts": contrasts(c, args.boot, rng),
              "subsets": subsets(c, args.boot, rng), "checks": checks(df, c)}
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500, "display.float_format", "{:.2f}".format)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
        if name == "overview": t = t[t.lang == "all"].drop(columns="lang")
        if name == "strata": t = t[(t.kind == "control") | (t.condition_id == DEFAULT)]      # full table in strata.csv
        print(f"\n== {name} ({args.split}, judge {df.judge_model.iloc[0]}, {args.judge_version}) ==\n{t.to_string(index=False)}")
    print(f"\nwrote {len(tables)} csv files to {out}")

if __name__ == "__main__":
    main()