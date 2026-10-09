"""LLM judge for the L4 decoding experiment: do two explanations of a coded phrase mean the same?
Symmetric pairwise comparison (Explanation A / B, order fixed per case and balanced within split and language), blind to condition and model.
  python judge.py run --results results/<name>.jsonl --judge-model <model> --split dev|test|all [--base-url URL] [--runs 1] [--dry-run]
  python judge.py control --kind literal|shuffled --judge-model <model> --split dev|test|all [--dry-run]
  python judge.py sample --judgments judgments/ctx-*__j2.jsonl [--n 100]   # blind sheet for hand labelling (dev only)
  python judge.py agree [--dir judgments]                                   # judge-vs-human agreement after labelling
Any OpenAI-compatible endpoint works (vLLM, OpenRouter, ...); key in env JUDGE_API_KEY. --no-schema if it lacks json_schema support;
--extra-body passes provider-specific fields; every judgment records the served model/provider and token counts.
TODO: the unnamed (detection ablation) conditions are skipped here; they get a separate standalone script (score_unnamed.py)."""
import argparse, collections, csv, fcntl, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import config

JUDGE_VERSION = "j2"                                            # j1 = asymmetric reference/candidate prompt (retired)
VERDICTS = ["not_equivalent", "partially_equivalent", "equivalent"]   # ordinal, low -> high
SCHEMA = {"type": "object", "required": ["reason", "verdict"], "additionalProperties": False,
          "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": VERDICTS}}}
RUBRIC = """You compare two explanations of the same coded expression from Russian or Turkish political journalism. Decide whether they convey the same insider meaning.
- equivalent: same core meaning (same referent, allusion, saying, event or concept); wording and extra detail may differ.
- partially_equivalent: overlapping meaning or the same general direction, but the core meaning differs or only part is shared.
- not_equivalent: different or unrelated meanings.
Judge the meaning of the expression only, not length or style. Neither explanation is a gold standard; either may be incomplete.

Examples (development set):
Expression: «Z-патриоты» | Explanation A: Supporters of Russia's invasion and pro-war propaganda. | Explanation B: hardline pro-war supporters (likely a political faction or label in Russian media) -> equivalent
Expression: «в формате письма Деду Морозу» | Explanation A: a mock-serious, ironic letter style used to draw attention to a serious issue | Explanation B: A childish, implausible wish addressed to a mythical gift-giver. -> partially_equivalent (shared ironic direction, A misses the childish, implausible wish)
Expression: «бояре плохие» | Explanation A: part of the saying "the Tsar is good, the boyars are bad"; boyars are the ruling class around a tsar | Explanation B: bad hussars, a pejorative term for incompetent military units -> not_equivalent

Answer with one JSON object: {"reason": "<one sentence, at most 30 words>", "verdict": "equivalent" | "partially_equivalent" | "not_equivalent"}"""
ANCHOR_PHRASES = {"Z-патриоты", "в формате письма Деду Морозу", "бояре плохие"}   # used in the rubric; excluded from validation

def load_refs():
    from datasets import load_from_disk
    return {r["case_id"]: r for r in load_from_disk(str(config.DS_DIR / "_references"))}

def load_results(path, runs):
    """Rows of a results dataset dir or .jsonl; `run` = occurrence index of the case_id in the file."""
    p = Path(path)
    if p.suffix == ".jsonl": rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    else:
        from datasets import load_from_disk
        rows = [dict(r) for r in load_from_disk(str(p))]
    seen, out = collections.Counter(), []
    for r in rows:
        r["run"] = seen[r["case_id"]]; seen[r["case_id"]] += 1
        if r["run"] < runs: out.append(r)
    if any(v > 1 for v in seen.values()):
        print(f"note: {sum(v > 1 for v in seen.values())} case_ids occur more than once (repeat runs); keeping the first {runs}")
    return out

def tasks_from_results(rows, refs, split):
    """Named conditions only: one model explanation per case, compared with the annotator's explanation."""
    tasks = []
    for r in rows:
        c = refs[r["case_id"]]
        if split != "all" and c["split"] != split: continue
        t = dict(case_id=r["case_id"], condition_id=r["condition_id"], model=r["model"], run=r["run"],
                 naming="named", candidate=None, matched_phrase="", verdict=None)
        try: out = json.loads(r["raw"]) if r["parse_ok"] else None
        except Exception: out = None
        t["candidate"] = str(out.get("insider", "")).strip() if isinstance(out, dict) else ""
        if not t["candidate"]: t["verdict"] = "invalid"
        tasks.append(t)
    return sorted(tasks, key=lambda t: (t["case_id"], t["run"]))

def control_tasks(kind, refs, split):
    """literal: the annotator's own literal reading as candidate (floor). shuffled: another case's reference (chance)."""
    rng, tasks = random.Random(config.SEED), []
    for lang in ("ru", "tr"):
        ids = sorted(k for k, c in refs.items() if c["lang"] == lang and (split == "all" or c["split"] == split))
        perm = ids[:]
        for _ in range(1000):                                   # reshuffle until nobody is paired with their own phrase
            rng.shuffle(perm)
            if all(refs[a]["phrase"].casefold() != refs[b]["phrase"].casefold() for a, b in zip(ids, perm)): break
        for a, b in zip(ids, perm):
            cand = (refs[a]["literal"] if kind == "literal" else refs[b]["insider"]).strip()
            tasks.append(dict(case_id=a, condition_id=f"control-{kind}", model="control", run=0, naming="named",
                              candidate=cand or None, matched_phrase="", verdict=None if cand else "invalid"))
    return tasks

def order_map(refs):
    """case_id -> True if the annotator's explanation is Explanation A. Exactly half of every (split, language) group,
    chosen by a seeded random ranking: balanced within dev and test, fixed across models, conditions and runs."""
    groups, order = collections.defaultdict(list), {}
    for k, c in refs.items(): groups[(c["split"], c["lang"])].append(k)
    for ids in groups.values():
        for i, k in enumerate(sorted(ids, key=lambda k: random.Random(f"{config.SEED}:{k}").random())): order[k] = i % 2 == 0
    return order

def user_message(c, candidate, ref_first):
    a, b = (c["insider"], candidate) if ref_first else (candidate, c["insider"])
    return f"Expression ({config.LANG_NAME[c['lang']]}): {c['phrase']}\nExplanation A: {a}\nExplanation B: {b}"

def dry_verdict(ref, cand):                                     # token overlap, only to test the plumbing
    a, b = set(ref.casefold().split()), set(cand.casefold().split())
    j = len(a & b) / max(1, len(a | b)); return ("equivalent" if j >= .4 else "partially_equivalent" if j >= .15 else "not_equivalent"), "dry-run heuristic"

def run_tasks(tasks, refs, args, out_path):
    out_path.parent.mkdir(exist_ok=True)
    lock = open(str(out_path) + ".lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(f"another judge run is writing {out_path}; refusing to start")
    done = {(r["case_id"], r["run"]) for r in map(json.loads, open(out_path)) if r["verdict"] != "judge_error"} if out_path.exists() else set()
    todo = [t for t in tasks if (t["case_id"], t["run"]) not in done]
    extra = json.loads(args.extra_body) if args.extra_body else {}          # fail fast on malformed JSON
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=os.environ.get("JUDGE_API_KEY", "none"))

    order = order_map(refs)

    def work(t):
        c, ref_first = refs[t["case_id"]], order[t["case_id"]]
        row = {**t, "judge_model": "dry-run" if args.dry_run else args.judge_model, "judge_version": JUDGE_VERSION,
               "reason": "", "a_is": "reference" if ref_first else "candidate",
               "served_model": "", "served_provider": "", "tokens_in": None, "tokens_out": None}
        if t["verdict"] is not None: return row                   # invalid outputs need no judge call
        if args.dry_run: row["verdict"], row["reason"] = dry_verdict(c["insider"], t["candidate"]); return row
        kw = dict(model=args.judge_model, temperature=0, max_tokens=args.max_tokens,
                  messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": user_message(c, t["candidate"], ref_first)}])
        if extra: kw["extra_body"] = extra
        if not args.no_schema: kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True}}
        err = ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kw); choice = resp.choices[0]
                m = re.search(r"\{.*\}", choice.message.content or "", re.S)
                if not m: err = f"no JSON in reply (finish_reason={choice.finish_reason})"
                else:
                    j = json.loads(m.group(0)); v = str(j["verdict"]).strip().lower()
                    if v in VERDICTS:
                        u = resp.usage
                        row.update(verdict=v, reason=str(j.get("reason", "")), served_model=resp.model,
                                   served_provider=str(getattr(resp, "provider", "") or ""),
                                   tokens_in=u.prompt_tokens if u else None, tokens_out=u.completion_tokens if u else None)
                        return row
                    err = f"bad verdict {j['verdict']!r}"
            except Exception as e: err = str(e)
            time.sleep(2 ** attempt)
        row["verdict"], row["reason"] = "judge_error", err; return row

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(work, todo):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    rows = [json.loads(l) for l in open(out_path)]
    n = len(rows); cnt = collections.Counter(r["verdict"] for r in rows)
    who = collections.Counter((r.get("served_model", ""), r.get("served_provider", "")) for r in rows if r.get("served_model"))
    if who: print("answered by:", dict(who), "| tokens in/out:", sum(r["tokens_in"] or 0 for r in rows), "/", sum(r["tokens_out"] or 0 for r in rows))
    print(f"{out_path.name}: {n} judgments ({len(todo)} new) | " + " | ".join(f"{v} {cnt[v]} ({cnt[v]/n:.0%})" for v in VERDICTS + ["invalid", "judge_error"] if cnt[v]))

def cmd_run(args):
    refs, rows = load_refs(), load_results(args.results, args.runs)
    if rows and rows[0]["condition_id"].endswith("__unnamed"):
        print(f"skipping {rows[0]['condition_id']}: unnamed (detection ablation) conditions are not judged here (TODO: score_unnamed.py)"); return
    name = Path(args.results).name if Path(args.results).suffix != ".jsonl" else Path(args.results).stem
    run_tasks(tasks_from_results(rows, refs, args.split), refs, args, config.JUDGE_DIR / f"{name}__{tag(args)}.jsonl")

def cmd_control(args):
    refs = load_refs()
    run_tasks(control_tasks(args.kind, refs, args.split), refs, args, config.JUDGE_DIR / f"control-{args.kind}__{args.split}__{tag(args)}.jsonl")

def tag(args): return f"{'dry-run' if args.dry_run else args.judge_model.replace('/', '_')}__{JUDGE_VERSION}"

def cmd_sample(args):
    """Blind labelling sheet (dev only, anchors excluded), balanced over the judge's verdicts; the key is kept separately."""
    refs, rows = load_refs(), []
    for f in args.judgments: rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["verdict"] in VERDICTS and r["candidate"] and refs[r["case_id"]]["split"] == "dev"
            and refs[r["case_id"]]["phrase"] not in ANCHOR_PHRASES]
    rng, by, pick = random.Random(config.SEED), collections.defaultdict(list), []
    for r in rows: by[r["verdict"]].append(r)
    for v in VERDICTS: rng.shuffle(by[v]); pick += by[v][:args.n // 3]
    rest = [r for r in rows if r not in pick]; rng.shuffle(rest); pick += rest[:args.n - len(pick)]
    rng.shuffle(pick); out = Path(args.dir); out.mkdir(exist_ok=True)
    with open(out / "label_sheet.csv", "w", newline="", encoding="utf-8") as fs, open(out / "label_key.csv", "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["id", "lang", "phrase", "explanation_a", "explanation_b", "human_verdict"])
        wk.writerow(["id", "case_id", "run", "condition_id", "a_is", "judge_verdict"])
        for i, r in enumerate(pick):
            c = refs[r["case_id"]]; a, b = (c["insider"], r["candidate"]) if r["a_is"] == "reference" else (r["candidate"], c["insider"])
            ws.writerow([f"s{i:03d}", c["lang"], c["phrase"], a, b, ""])
            wk.writerow([f"s{i:03d}", r["case_id"], r["run"], r["condition_id"], r["a_is"], r["verdict"]])
    print(f"wrote {len(pick)} rows to {out}/label_sheet.csv (fill human_verdict with {'/'.join(VERDICTS)}) and label_key.csv (judge verdicts, do not open)")

def kappa(x, y, linear=False, k=3):
    O = np.zeros((k, k))
    for i, j in zip(x, y): O[i, j] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0))
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1) if linear else 1 - np.eye(k)
    return 1 - (W * O).sum() / (W * E).sum()

def cmd_agree(args):
    import pandas as pd
    s, k = pd.read_csv(Path(args.dir) / "label_sheet.csv"), pd.read_csv(Path(args.dir) / "label_key.csv")
    d = s.merge(k, on="id"); bad = d[d.human_verdict.notna() & ~d.human_verdict.isin(VERDICTS)]
    if len(bad): sys.exit(f"{len(bad)} human_verdict values are not one of {VERDICTS}, e.g. {bad.human_verdict.iloc[0]!r}")
    d = d[d.human_verdict.isin(VERDICTS)]
    h, j = d.human_verdict.map(VERDICTS.index), d.judge_verdict.map(VERDICTS.index)
    print(f"n = {len(d)} | agreement {np.mean(h == j):.1%} | Cohen kappa {kappa(h, j):.2f} | linear-weighted kappa {kappa(h, j, True):.2f}")
    print("rows = human, columns = judge\n", pd.crosstab(d.human_verdict, d.judge_verdict).reindex(index=VERDICTS, columns=VERDICTS, fill_value=0))
    d = d.assign(h=h, j=j, group=np.where(d.condition_id.str.startswith("control-"), "control items", "model outputs"))
    print(f"judge more lenient than human: {np.mean(d.j > d.h):.0%} | stricter: {np.mean(d.j < d.h):.0%}")
    for g, x in d.groupby("group"): print(f"  {g}: n={len(x)} | agreement {np.mean(x.h == x.j):.0%} | lenient {np.mean(x.j > x.h):.0%} | stricter {np.mean(x.j < x.h):.0%}")

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--judge-model"); p.add_argument("--base-url"); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--split", required=True, choices=["dev", "test", "all"]); p.add_argument("--dry-run", action="store_true")
        p.add_argument("--no-schema", action="store_true"); p.add_argument("--max-tokens", type=int, default=200)
        p.add_argument("--extra-body", help='JSON merged into the request, e.g. OpenRouter: \'{"reasoning": {"effort": "none"}, "provider": {"order": ["openai"], "allow_fallbacks": false}}\'')
    r = sub.add_parser("run"); common(r); r.add_argument("--results", required=True); r.add_argument("--runs", type=int, default=1); r.set_defaults(f=cmd_run)
    c = sub.add_parser("control"); common(c); c.add_argument("--kind", required=True, choices=["literal", "shuffled"]); c.set_defaults(f=cmd_control)
    s = sub.add_parser("sample"); s.add_argument("--judgments", nargs="+", required=True); s.add_argument("--n", type=int, default=100)
    s.add_argument("--dir", default=str(config.JUDGE_DIR)); s.set_defaults(f=cmd_sample)
    a = sub.add_parser("agree"); a.add_argument("--dir", default=str(config.JUDGE_DIR)); a.set_defaults(f=cmd_agree)
    args = ap.parse_args()
    if args.cmd in ("run", "control") and not args.dry_run and not args.judge_model: ap.error("--judge-model is required unless --dry-run")
    args.f(args)

if __name__ == "__main__":
    main()