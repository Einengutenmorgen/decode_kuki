#!/usr/bin/env bash
# Judge every named condition of one evaluated model, then the controls. Resumable: finished cases are skipped.
# usage: [MODEL=olmo3-7b-instruct] [WORKERS=6] ./judge_all.sh <dev|test|all> [runs]     (needs JUDGE_API_KEY = OpenRouter key)
#        DRY=1 ./judge_all.sh dev            -> plumbing test without any API call

# j3: every item gets two judge calls (insider round + literal round); controls = shuffled, literal, reference.
# Run the splits separately, not `all`: the shuffled control pairs cases within the requested split, so dev and `all` would
# write conflicting pairings for the same dev cases.
#   dev:   ./judge_all.sh dev
#   test:  ./judge_all.sh test 2        (only after the rubric is frozen; runs=2 only matters if the results files hold repeat runs)
#   check: DRY=1 ./judge_all.sh dev     (writes dry-run__j3 files; pass --judge-model to analysis.py afterwards)

set -euo pipefail
SPLIT=${1:?usage: ./judge_all.sh <dev|test|all> [runs]}; RUNS=${2:-1}; MODEL=${MODEL:-*}  
EXTRA='{"reasoning":{"effort":"none"},"provider":{"order":["openai"],"allow_fallbacks":false}}'
if [ "${DRY:-0}" = 1 ]; then J=(--dry-run)
else J=(--judge-model openai/gpt-6-luna-20260922 --base-url https://openrouter.ai/api/v1 --max-tokens 600 --workers "${WORKERS:-6}" --extra-body "$EXTRA"); fi
for r in results_*/*__named__${MODEL}__v2.jsonl; do                     # named conditions only; *__unnamed__* does not match
  python judge.py run --results "$r" --split "$SPLIT" --runs "$RUNS" "${J[@]}"
done
for kind in shuffled literal reference; do python judge.py control --kind "$kind" --split "$SPLIT" "${J[@]}"; done