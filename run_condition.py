"""Run one condition with one model against an OpenAI-compatible server (e.g. `vllm serve <hf_id>`).
Resumable: finished case_ids in results/<condition>__<model>.jsonl are skipped.
usage: python run_condition.py --model gemma4-12b --condition ctx-paragraph__title-0__outlet-none__en__named
       [--split dev|test|all] [--limit N] [--base-url http://localhost:8000/v1] [--dry-run]"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, load_from_disk
import config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=config.MODELS)
ap.add_argument("--condition", required=True)
ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])   # dev first: prompts are frozen on dev only
ap.add_argument("--limit", type=int)
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--dry-run", action="store_true", help="no server: echo the phrase, tests the plumbing")
args = ap.parse_args()

rows = [r for r in load_from_disk(str(config.DS_DIR / args.condition)) if args.split == "all" or r["split"] == args.split]
rows = rows[:args.limit] if args.limit else rows
mcfg = config.MODELS[args.model]
tag = f"{args.condition}__{args.model}__{rows[0]['prompt_version']}"      # prompt version keeps results of different prompts apart
out_path = config.RES_DIR / f"{tag}.jsonl"; config.RES_DIR.mkdir(exist_ok=True)
done = {json.loads(l)["case_id"] for l in open(out_path)} if out_path.exists() else set()
todo = [r for r in rows if r["case_id"] not in done]

if args.dry_run:
    ask = lambda r: json.dumps({"literal": r["phrase"], "insider": "dry run", "why": "dry run"})
else:
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="none")
    def ask(r):                                                         # greedy decoding, schema-constrained JSON
        resp = client.chat.completions.create(
            model=mcfg["hf_id"], messages=r["messages"], temperature=0,
            max_tokens=400 if r["naming"] == "named" else 1200,
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": json.loads(r["schema"]), "strict": True}},
            extra_body=mcfg["extra_body"])
        return resp.choices[0].message.content

def work(r):
    try: raw = ask(r)
    except Exception as e: raw = f"ERROR: {e}"
    try: json.loads(raw); ok = True
    except Exception: ok = False
    return {"case_id": r["case_id"], "condition_id": r["condition_id"], "model": args.model, "raw": raw, "parse_ok": ok}

t0 = time.time()
with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
    for res in pool.map(work, todo):
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()

all_res = [json.loads(l) for l in open(out_path)]
Dataset.from_list(all_res).save_to_disk(str(config.RES_DIR / tag))
errors = sum(r["raw"].startswith("ERROR") for r in all_res)
print(f"{args.condition} / {args.model} / {rows[0]['prompt_version']}: {len(all_res)} results ({len(todo)} new), "
      f"parse_ok {sum(r['parse_ok'] for r in all_res)}, errors {errors}, {len(todo) / max(time.time() - t0, 1e-9):.2f} cases/s")