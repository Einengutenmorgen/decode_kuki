"""LLM judge for the L4 decoding experiment: do two explanations of a coded phrase mean the same?
Two rounds of calls per item (j3):
  1. insider round: symmetric pairwise comparison (Explanation A / B, order fixed per case and balanced within split and language),
     blind to condition and model -> verdict (core meaning), plus which side holds substantial extra ideas (mapped back to the candidate).
  2. literal round: how far does the candidate only restate the annotator's literal (surface) reading? -> literal_overlap.
  python judge.py run --results results/<n>.jsonl --judge-model <model> --split dev|test|all [--base-url URL] [--runs 1] [--dry-run]
  python judge.py control --kind literal|shuffled|reference|annotator --judge-model <model> --split dev|test|all [--dry-run]
      literal = annotator's literal reading as candidate (floor); shuffled = another case's reference (chance);
      reference = the annotator's own insider text as candidate (identity check; round 2 shows which references only restate their literal reading)
      annotator = the insider text of another annotator who marked the same span in the same article (human ceiling; test split only, written to judgments/annotator/)
  python judge.py sample --judgments judgments/ctx-*__j3.jsonl [--n 100]   # blind sheet for hand labelling (dev only)
  python judge.py agree [--dir judgments]                                   # judge-vs-human agreement after labelling
Any OpenAI-compatible endpoint works (vLLM, OpenRouter, ...); key in env JUDGE_API_KEY. --no-schema if it lacks json_schema support;
--extra-body passes provider-specific fields; every judgment records the served model/provider and token counts (summed over both rounds).
TODO: the unnamed (detection ablation) conditions are skipped here; they get a separate standalone script (score_unnamed.py)."""
import argparse, collections, csv, fcntl, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import config

JUDGE_VERSION = "j3"                                            # j1 = asymmetric reference/candidate prompt (retired); j2 = one call, verdict only
VERDICTS = ["not_equivalent", "partially_equivalent", "equivalent"]   # ordinal, low -> high
SCHEMA = {"type": "object", "required": ["reason", "verdict", "only_a", "only_b"], "additionalProperties": False,
          "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": VERDICTS},
                         "only_a": {"type": "boolean"}, "only_b": {"type": "boolean"}}}
LITERAL_SCHEMA = {"type": "object", "required": ["reason", "literal_overlap"], "additionalProperties": False,
                  "properties": {"reason": {"type": "string"}, "literal_overlap": {"type": "string", "enum": VERDICTS}}}
RUBRIC = """You compare two explanations of the same coded expression from Russian or Turkish political journalism. Decide whether they convey the same insider meaning.
The core meaning is what the expression signals to an in-group reader (its stance, allusion, implication, or the saying, event or person it points to), not merely what it names or translates to.
- equivalent: same core meaning; wording and extra detail may differ.
- partially_equivalent: overlapping meaning or the same general direction, but the core meaning differs or only part is shared. An explanation that only names or translates the expression, without saying what it signals, is at most partially_equivalent when the other explanation does say what it signals.
- not_equivalent: different or unrelated meanings.
Judge the meaning of the expression only, not length or style. Neither explanation is a gold standard; either may be incomplete.
Then record distinct ideas beyond what the two share: only_a = true if Explanation A makes a distinct claim about who, what or why that B does not make (an additional or different referent, motive, stance or implication); only_b = true likewise for B. Elaboration, dates, examples, synonyms, more precise wording, or a more explicit version of the same idea do not count. Both are false when the verdict is not_equivalent.

Examples:
Expression: «Z-патриоты» | Explanation A: Supporters of Russia's invasion and pro-war propaganda. | Explanation B: hardline pro-war supporters (likely a political faction or label in Russian media) -> equivalent; only_a false, only_b false
Expression: «в формате письма Деду Морозу» | Explanation A: a mock-serious, ironic letter style used to draw attention to a serious issue | Explanation B: A childish, implausible wish addressed to a mythical gift-giver. -> partially_equivalent (shared ironic direction); only_a true (draws attention to a serious issue), only_b true (childish, implausible wish)
Expression: «бояре плохие» | Explanation A: part of the saying "the Tsar is good, the boyars are bad"; boyars are the ruling class around a tsar | Explanation B: bad hussars, a pejorative term for incompetent military units -> not_equivalent; only_a false, only_b false
Expression: «синие занавески» | Explanation A: The blue curtains. | Explanation B: A conventional euphemism for covert surveillance of critics by the security services. -> partially_equivalent (A only names the expression, B says what it signals); only_a false, only_b true (the surveillance meaning)

Answer with one JSON object: {"reason": "<two sentences, at most 50 words: what the explanations share, what differs>", "verdict": "equivalent" | "partially_equivalent" | "not_equivalent", "only_a": true | false, "only_b": true | false}"""
LITERAL_RUBRIC = """An expression from Russian or Turkish political journalism has a plain surface rendering (what an outsider would understand from the words alone) and an explanation of what it means. Decide how far the explanation only restates that surface rendering.
- equivalent: the explanation only restates or paraphrases the surface rendering; it adds no meaning beyond the words.
- partially_equivalent: it restates the surface rendering and adds some further meaning.
- not_equivalent: it conveys a meaning clearly beyond the surface rendering, or is unrelated to it.

Examples:
Expression: «чайник на плите» | Surface rendering: the kettle on the stove | Explanation: A kettle that is standing on a stove. -> equivalent
Expression: «в формате письма Деду Морозу» | Surface rendering: in the format of a letter to Grandfather Frost | Explanation: A childish, implausible wish addressed to a mythical gift-giver. -> partially_equivalent (keeps the letter-to-a-gift-giver image, adds the implausible wish)
Expression: «Z-патриоты» | Surface rendering: Z-patriots | Explanation: Supporters of Russia's invasion and pro-war propaganda. -> not_equivalent

Answer with one JSON object: {"reason": "<one sentence, at most 30 words>", "literal_overlap": "equivalent" | "partially_equivalent" | "not_equivalent"}"""
ANCHOR_PHRASES = {"Z-патриоты", "в формате письма Деду Морозу", "бояре плохие"}   # used in the rubrics; excluded from validation

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
    """literal: the annotator's own literal reading as candidate (floor). shuffled: another case's reference (chance).
    reference: the annotator's own insider text as candidate (identity check; round 2 flags references that only restate their literal reading).
    annotator: the insider text of another annotator who marked the same span in the same article (human ceiling). One task per ordered
    pair: the case is the reference row, `run` counts its partners, `partner_id` names the candidate row."""
    rng, tasks = random.Random(config.SEED), []
    if kind == "annotator":
        spans = collections.defaultdict(list)                   # (article, span "start-end") -> case_ids of all annotators
        for k, c in sorted(refs.items()):
            if split == "all" or c["split"] == split: spans[(c["article_id"], k.rsplit(":", 1)[-1])].append(k)
        for ids in spans.values():
            for a in ids:
                for i, b in enumerate(x for x in ids if refs[x]["annotator"] != refs[a]["annotator"]):
                    cand = refs[b]["insider"].strip()
                    tasks.append(dict(case_id=a, condition_id="control-annotator", model="control", run=i, naming="named",
                                      candidate=cand or None, matched_phrase="", partner_id=b, verdict=None if cand else "invalid"))
        print(f"annotator pairs: {len(tasks)} ordered pairs over {len({t['case_id'] for t in tasks})} reference rows ({split})")
        return tasks
    for lang in ("ru", "tr"):
        ids = sorted(k for k, c in refs.items() if c["lang"] == lang and (split == "all" or c["split"] == split))
        perm = ids[:]
        for _ in range(1000):                                   # reshuffle until nobody is paired with their own phrase
            rng.shuffle(perm)
            if all(refs[a]["phrase"].casefold() != refs[b]["phrase"].casefold() for a, b in zip(ids, perm)): break
        for a, b in zip(ids, perm):
            cand = (refs[a]["literal"] if kind == "literal" else refs[a]["insider"] if kind == "reference" else refs[b]["insider"]).strip()
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

def literal_message(c, candidate):                              # roles differ (surface rendering vs explanation), so no A/B order
    return f"Expression ({config.LANG_NAME[c['lang']]}): {c['phrase']}\nSurface rendering: {c['literal']}\nExplanation: {candidate}"

def check_main(j):
    """Validate/normalise the insider-round reply in place; '' = ok, else an error string (the call is retried)."""
    j["verdict"] = str(j.get("verdict", "")).strip().lower()
    if j["verdict"] not in VERDICTS: return f"bad verdict {j['verdict']!r}"
    for k in ("only_a", "only_b"):
        if not isinstance(j.get(k), bool): return f"bad {k} {j.get(k)!r}"
    return ""

def check_literal(j):
    j["literal_overlap"] = str(j.get("literal_overlap", "")).strip().lower()
    return "" if j["literal_overlap"] in VERDICTS else f"bad literal_overlap {j['literal_overlap']!r}"

def ask(client, args, extra, system, user, schema, check):
    """One judge call with up to 3 attempts -> (reply dict, response, '') or (None, None, error)."""
    kw = dict(model=args.judge_model, temperature=0, max_tokens=args.max_tokens,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if extra: kw["extra_body"] = extra
    if not args.no_schema: kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "judgment", "schema": schema, "strict": True}}
    err = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kw); choice = resp.choices[0]
            m = re.search(r"\{.*\}", choice.message.content or "", re.S)
            if not m: err = f"no JSON in reply (finish_reason={choice.finish_reason})"
            else:
                j = json.loads(m.group(0)); err = check(j)
                if not err: return j, resp, ""
        except Exception as e: err = str(e)
        time.sleep(2 ** attempt)
    return None, None, err

def add_tokens(row, resp):
    u = resp.usage
    if u: row["tokens_in"] = (row["tokens_in"] or 0) + u.prompt_tokens; row["tokens_out"] = (row["tokens_out"] or 0) + u.completion_tokens

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
               "reason": "", "extra_content": None, "missing_content": None,    # candidate vs annotator explanation, from round 1
               "literal_overlap": None, "literal_reason": "",                   # round 2
               "a_is": "reference" if ref_first else "candidate",
               "served_model": "", "served_provider": "", "tokens_in": None, "tokens_out": None}
        if t["verdict"] is not None: return row                   # invalid outputs need no judge call
        if args.dry_run:
            row["verdict"], row["reason"] = dry_verdict(c["insider"], t["candidate"])
            row.update(extra_content=False, missing_content=False, literal_reason="dry-run heuristic",
                       literal_overlap=dry_verdict(c["literal"] or "", t["candidate"])[0]); return row

        j, resp, err = ask(client, args, extra, RUBRIC, user_message(c, t["candidate"], ref_first), SCHEMA, check_main)
        if j is None: row["verdict"], row["reason"] = "judge_error", err; return row
        only_cand, only_ref = (j["only_b"], j["only_a"]) if ref_first else (j["only_a"], j["only_b"])
        diff = j["verdict"] != "not_equivalent"                   # extra/missing detail only counts on a shared core
        row.update(verdict=j["verdict"], reason=str(j.get("reason", "")), extra_content=only_cand and diff, missing_content=only_ref and diff,
                   served_model=resp.model, served_provider=str(getattr(resp, "provider", "") or ""))
        add_tokens(row, resp)

        if (c["literal"] or "").strip():
            j, resp, err = ask(client, args, extra, LITERAL_RUBRIC, literal_message(c, t["candidate"]), LITERAL_SCHEMA, check_literal)
            if j is None: row["verdict"], row["reason"] = "judge_error", f"literal round: {err}"; return row   # redone on the next run
            row.update(literal_overlap=j["literal_overlap"], literal_reason=str(j.get("reason", "")))
            add_tokens(row, resp)
        return row

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(work, todo):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    rows = [json.loads(l) for l in open(out_path)]
    n = len(rows); cnt = collections.Counter(r["verdict"] for r in rows)
    who = collections.Counter((r.get("served_model", ""), r.get("served_provider", "")) for r in rows if r.get("served_model"))
    if who: print("answered by:", dict(who), "| tokens in/out:", sum(r["tokens_in"] or 0 for r in rows), "/", sum(r["tokens_out"] or 0 for r in rows))
    print(f"{out_path.name}: {n} judgments ({len(todo)} new) | " + " | ".join(f"{v} {cnt[v]} ({cnt[v]/n:.0%})" for v in VERDICTS + ["invalid", "judge_error"] if cnt[v]))
    lit = collections.Counter(r.get("literal_overlap") for r in rows if r.get("literal_overlap"))
    if lit: print("  literal_overlap: " + " | ".join(f"{v} {lit[v]}" for v in VERDICTS if lit[v])
                  + f" | extra_content {sum(bool(r.get('extra_content')) for r in rows)} | missing_content {sum(bool(r.get('missing_content')) for r in rows)}")

def cmd_run(args):
    refs, rows = load_refs(), load_results(args.results, args.runs)
    if rows and rows[0]["condition_id"].endswith("__unnamed"):
        print(f"skipping {rows[0]['condition_id']}: unnamed (detection ablation) conditions are not judged here (TODO: score_unnamed.py)"); return
    name = Path(args.results).name if Path(args.results).suffix != ".jsonl" else Path(args.results).stem
    run_tasks(tasks_from_results(rows, refs, args.split), refs, args, config.JUDGE_DIR / f"{name}__{tag(args)}.jsonl")

def cmd_control(args):
    refs = load_refs()
    sub = "annotator" if args.kind == "annotator" else ""      # pair rows repeat case_ids: own folder, so analysis.py's glob does not see them
    run_tasks(control_tasks(args.kind, refs, args.split), refs, args, config.JUDGE_DIR / sub / f"control-{args.kind}__{args.split}__{tag(args)}.jsonl")

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
        p.add_argument("--no-schema", action="store_true"); p.add_argument("--max-tokens", type=int, default=400)
        p.add_argument("--extra-body", help='JSON merged into the request, e.g. OpenRouter: \'{"reasoning": {"effort": "none"}, "provider": {"order": ["openai"], "allow_fallbacks": false}}\'')
    r = sub.add_parser("run"); common(r); r.add_argument("--results", required=True); r.add_argument("--runs", type=int, default=1); r.set_defaults(f=cmd_run)
    c = sub.add_parser("control"); common(c); c.add_argument("--kind", required=True, choices=["literal", "shuffled", "reference", "annotator"]); c.set_defaults(f=cmd_control)
    s = sub.add_parser("sample"); s.add_argument("--judgments", nargs="+", required=True); s.add_argument("--n", type=int, default=100)
    s.add_argument("--dir", default=str(config.JUDGE_DIR)); s.set_defaults(f=cmd_sample)
    a = sub.add_parser("agree"); a.add_argument("--dir", default=str(config.JUDGE_DIR)); a.set_defaults(f=cmd_agree)
    args = ap.parse_args()
    if args.cmd in ("run", "control") and not args.dry_run and not args.judge_model: ap.error("--judge-model is required unless --dry-run")
    args.f(args)

if __name__ == "__main__":
    main()