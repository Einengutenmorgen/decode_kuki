"""Build one Hugging Face dataset per condition (inputs only) plus a references dataset.
usage: python build_datasets.py"""
import json, random, re, shutil
from collections import defaultdict
from datasets import Dataset
import config, prompts

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def paragraphs(content, lang):
    """(start, end) of non-empty paragraphs. The first one is the title block."""
    sep, out, pos = config.PARA_SEP[lang], [], 0
    for part in content.split(sep):
        if part.strip(): out.append((pos, pos + len(part)))
        pos += len(part) + len(sep)
    return out

_END = re.compile(r'([.!?…]+["»”’)\]]*)\s+(?=["«“(\[]?[A-ZА-ЯЁÇĞİÖŞÜ0-9])')
_ABBR = {"г", "гг", "ул", "д", "им", "др", "см", "т", "тыс", "млн", "млрд", "руб", "dr", "prof", "no", "bkz", "örn", "vb"}

def sentences(content, paras):
    """Rule-based splitter inside paragraphs; skips initials ("В. В. Путин") and common abbreviations."""
    out = []
    for a, b in paras:
        start = a
        for m in _END.finditer(content, a, b):
            word = re.search(r"(\w+)$", content[max(a, m.start(1) - 12):m.start(1)])
            w = word.group(1) if word else ""
            if (len(w) == 1 and w.isupper()) or w.lower() in _ABBR: continue
            out.append((start, m.end(1))); start = m.end()
        out.append((start, b))
    return out

def overlapping(ranges, s, e):
    idx = [i for i, (a, b) in enumerate(ranges) if s < b and e > a]
    return idx or [max(i for i, (a, b) in enumerate(ranges) if a <= s)]

def mark(content, a, b, s, e):
    s, e = max(s, a), min(e, b)
    return content[a:s] + "[[" + content[s:e] + "]]" + content[e:b]

def load_cases():
    """One case per L4 span that has an insider-meaning explanation."""
    cases, skipped = [], 0
    for lang in ("ru", "tr"):
        for rec in read_jsonl(config.DATA_DIR / f"kuki_{lang}_aggregate.jsonl"):
            content = rec["content"]; paras = paragraphs(content, lang); sents = sentences(content, paras)
            for uid, ann in rec["annotations"].items():
                who = f"{lang}_{config.ANNOTATOR_IDS[lang][int(uid)]}"
                for sp in ann["spans"]:
                    if sp["layer"] != config.L4_LAYER: continue
                    if not (sp.get("insider") or "").strip(): skipped += 1; continue
                    s, e = sp["start"], sp["end"]; raw = content[s:e]
                    s, e = s + len(raw) - len(raw.lstrip()), e - (len(raw) - len(raw.rstrip()))
                    pi, si = overlapping(paras, s, e), overlapping(sents, s, e)
                    cases.append(dict(
                        case_id=f"{lang}:{rec['doc_id']}:{who}:{s}-{e}", lang=lang, article_id=rec["doc_id"],
                        source=rec["source"], annotator=who, phrase=content[s:e], start=s, end=e,
                        title=rec["title"].strip(), content=content,
                        par=(paras[pi[0]][0], paras[pi[-1]][1]), sent=(sents[si[0]][0], sents[si[-1]][1]),
                        span_in_title=pi[0] == 0, crosses_sentence=len(si) > 1, crosses_paragraph=len(pi) > 1,
                        literal=sp.get("literal", ""), insider=sp["insider"].strip(), why=sp.get("why", "")))
    print(f"{len(cases)} cases ({skipped} spans without insider explanation skipped)")
    return sorted(cases, key=lambda c: c["case_id"])

def add_flags_and_split(cases):
    norm = lambda c: (c["lang"], " ".join(c["phrase"].casefold().split()))
    who = defaultdict(set)
    for c in cases: who[norm(c)].add(c["annotator"])
    for c in cases:
        c["is_multiref"] = len(who[norm(c)]) >= 2          # phrase explained by >=2 annotators (variance set)
        c["same_article_multiref"] = any(o["article_id"] == c["article_id"] and o["annotator"] != c["annotator"]
                                         and o["start"] < c["end"] and o["end"] > c["start"] for o in cases
                                         if o["lang"] == c["lang"])
    rng, groups = random.Random(config.SEED), defaultdict(list)
    for c in cases:
        if not c["is_multiref"]: groups[norm(c)].append(c)
    keys = sorted(groups); rng.shuffle(keys); got = defaultdict(int)
    dev = set()
    for k in keys:                                          # phrase-grouped dev draw, variance set never in dev
        if got[k[0]] < config.DEV_SIZE[k[0]]:
            dev.update(c["case_id"] for c in groups[k]); got[k[0]] += len(groups[k])
    for c in cases: c["split"] = "dev" if c["case_id"] in dev else "test"

def swapped_outlets(cases):
    """article_id -> outlet of the opposite pole, assigned round-robin so every outlet is used evenly."""
    shown, counter = {}, defaultdict(int)
    for aid, src in sorted({(c["article_id"], c["source"]) for c in cases}):
        lang = "ru" if src in ("ria_novosti", "ng", "theinsider", "holod") else "tr"
        pole = config.OUTLETS[src][1]
        opposite = sorted(s for s, (_, p) in config.OUTLETS.items()
                          if p != pole and (s in ("ria_novosti", "ng", "theinsider", "holod")) == (lang == "ru"))
        shown[aid] = opposite[counter[(lang, pole)] % len(opposite)]; counter[(lang, pole)] += 1
    return shown

def context_text(c, level, marked):
    a, b = {"sentence": c["sent"], "paragraph": c["par"], "article": (0, len(c["content"]))}.get(level, (None, None))
    if a is None: return None
    return mark(c["content"], a, b, c["start"], c["end"]) if marked else c["content"][a:b]

def build_row(c, cd, shown):
    named = cd["naming"] == "named"
    outlet_src = {"none": None, "true": c["source"], "swapped": shown[c["article_id"]]}[cd["outlet"]]
    show_title = cd["title"] and not c["span_in_title"]     # the title is already the context for title spans
    messages, schema = prompts.render(
        cd["prompt_lang"], language=config.LANG_NAME[c["lang"]], phrase=c["phrase"],
        context=context_text(c, cd["context"], marked=named), naming=cd["naming"],
        outlet=config.OUTLETS[outlet_src][0] if outlet_src else "", title=c["title"] if show_title else "")
    return dict(case_id=c["case_id"], condition_id=config.condition_id(cd), lang=c["lang"], article_id=c["article_id"],
                source=c["source"], pole_true=config.OUTLETS[c["source"]][1], annotator=c["annotator"], split=c["split"],
                is_multiref=c["is_multiref"], span_in_title=c["span_in_title"], phrase=c["phrase"],
                context=cd["context"], title_shown=cd["title"], outlet_mode=cd["outlet"],
                outlet_shown=outlet_src or "", pole_shown=config.OUTLETS[outlet_src][1] if outlet_src else "",
                prompt_lang=cd["prompt_lang"], naming=cd["naming"], prompt_version=config.PROMPT_VERSION, messages=messages, schema=json.dumps(schema),
                n_chars=sum(len(m["content"]) for m in messages))

def main():
    cases = load_cases(); add_flags_and_split(cases); shown = swapped_outlets(cases)
    config.DS_DIR.mkdir(exist_ok=True)
    refs = [{k: c[k] for k in ("case_id", "lang", "article_id", "annotator", "phrase", "literal", "insider", "why",
                               "split", "is_multiref", "same_article_multiref", "span_in_title",
                               "crosses_sentence", "crosses_paragraph")} for c in cases]
    Dataset.from_list(refs).save_to_disk(str(config.DS_DIR / "_references"))
    for cd in config.CONDITIONS:
        rows = [build_row(c, cd, shown) for c in cases]
        out = config.DS_DIR / config.condition_id(cd); shutil.rmtree(out, ignore_errors=True)
        Dataset.from_list(rows).save_to_disk(str(out))
        print(f"{config.condition_id(cd):62s} {len(rows)} cases, median prompt {sorted(r['n_chars'] for r in rows)[len(rows)//2]} chars")

if __name__ == "__main__":
    main()