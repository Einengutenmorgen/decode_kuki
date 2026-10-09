#!/usr/bin/env bash
# Judge every named condition of one evaluated model, then the controls. Resumable: finished cases are skipped.
# usage: [MODEL=olmo3-7b-instruct] [WORKERS=6] ./judge_all.sh <dev|test|all> [runs]     (needs JUDGE_API_KEY = OpenRouter key)
#        DRY=1 ./judge_all.sh dev            -> plumbing test without any API call

# For dev check 
# DRY=1 ./judge_all.sh dev
# ./judge_all.sh dev

#full run make use of double run
# full run ./judge_all.sh all 2

set -euo pipefail
SPLIT=${1:?usage: ./judge_all.sh <dev|test|all> [runs]}; RUNS=${2:-1}; MODEL=${MODEL:-olmo3-7b-instruct}
EXTRA='{"reasoning":{"effort":"none"},"provider":{"order":["openai"],"allow_fallbacks":false}}'
if [ "${DRY:-0}" = 1 ]; then J=(--dry-run)
else J=(--judge-model openai/gpt-6-luna-20260922 --base-url https://openrouter.ai/api/v1 --max-tokens 600 --workers "${WORKERS:-6}" --extra-body "$EXTRA"); fi
for r in results/*__named__"$MODEL"__v2.jsonl; do                    # named conditions only; *__unnamed__* does not match
  python judge.py run --results "$r" --split "$SPLIT" --runs "$RUNS" "${J[@]}"
done
for kind in shuffled literal; do python judge.py control --kind "$kind" --split "$SPLIT" "${J[@]}"; done
