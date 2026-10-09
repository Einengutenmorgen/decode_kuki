#!/usr/bin/env bash
# Overnight generation for the remaining models.
# Per model: (1) is vLLM running? (2) if so unload it and delete its weights from the HF cache, (4) load the model, (5) run every condition,
# then unload and delete the cache again. Phase 1 = dev for all model x condition pairs; phase 2 = test, only if every dev pair passed.
# Restartable: finished cases and finished model/split pairs are skipped.   usage: ./overnight.sh
# env knobs: MODELS PHASES GATE(strict|lenient) PORT MAXLEN UTIL WORKERS MAX_PARSE_FAIL WAIT_MIN POLL SKIP_UNNAMED PY VLLM_BIN LOGDIR
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}; VLLM_BIN=${VLLM_BIN:-vllm}; ME=${USER:-$(id -un)}
MODELS=${MODELS:-"gemma4-e4b gemma4-12b qwen3.8-27b gemma4-31b olmo3.1-32b"}      # smallest first: failures show up early
PHASES=${PHASES:-"dev test"}; GATE=${GATE:-strict}
PORT=${PORT:-8000}; MAXLEN=${MAXLEN:-16384}; UTIL=${UTIL:-0.80}; WORKERS=${WORKERS:-16}
MAX_PARSE_FAIL=${MAX_PARSE_FAIL:-0.10}; WAIT_MIN=${WAIT_MIN:-45}; POLL=${POLL:-10}
LOGDIR=${LOGDIR:-logs/overnight_$(date +%Y%m%d_%H%M)}; mkdir -p "$LOGDIR"; STATUS="$LOGDIR/status.tsv"; : > "$STATUS"
# alias -> "GPUs|tensor-parallel|needs patched Olmo chat template (1/0)".  EDIT to the GPUs that are free (nvidia-smi).
declare -A GPUS=( [gemma4-e4b]="0|1|0" [gemma4-12b]="0|1|0" [qwen3.8-27b]="0,1|2|0" [gemma4-31b]="0,1|2|0" [olmo3.1-32b]="0,1|2|1" )
CURRENT_HF=""

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/overnight.log"; }
hf_id() { $PY -c "import config; print(config.MODELS['$1']['hf_id'])"; }
cache_root() { $PY -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"; }
served_model() { curl -s --max-time 5 "http://localhost:$PORT/v1/models" | $PY -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true; }
vllm_pids() { pgrep -u "$ME" -f "vllm serve|VLLM::EngineCore" | tr '\n' ' '; }

delete_cache() {                         # free the disk: remove one model from the HF cache
  local id=$1 root d; [ -n "$id" ] || return 0
  root=$(cache_root); d="$root/models--${id//\//--}"
  case "$d" in "$root"/models--*) ;; *) log "refusing to delete $d"; return 1;; esac
  if [ -d "$d" ]; then log "deleting cache $d ($(du -sh "$d" | cut -f1))"; rm -rf "$d" "$root/.locks/models--${id//\//--}"; else log "no cache to delete for $id"; fi
}

unload_all() {                           # steps 1+2: is vLLM running? then stop it and delete the cache of the model it served
  local m waited=0; m=$(served_model)
  [ -z "$(vllm_pids)" ] && [ -z "$m" ] && return 0
  log "vLLM is running${m:+ (model: $m)}; stopping (pids: $(vllm_pids))"
  pkill -TERM -u "$ME" -f "vllm serve"
  while [ -n "$(vllm_pids)" ] && [ $waited -lt 120 ]; do sleep 2; waited=$((waited+2)); done
  if [ -n "$(vllm_pids)" ]; then log "force-killing leftovers: $(vllm_pids)"; pkill -KILL -u "$ME" -f "vllm serve|VLLM::EngineCore"; sleep 3; fi
  [ -n "$m" ] && delete_cache "$m"
  return 0
}

cleanup() { trap - EXIT; log "cleanup: unloading server and deleting the current model's cache"; unload_all; [ -n "$CURRENT_HF" ] && delete_cache "$CURRENT_HF"; }
trap cleanup EXIT; trap 'exit 130' INT TERM

condition_order() {                      # named conditions (fast to slow) first, unnamed last: an interrupted run keeps the important ones
  local all; all=$(ls datasets | grep -v '^_')
  echo "$all" | grep -E '__named$' | grep -v '^ctx-article'; echo "$all" | grep -E '^ctx-article.*__named$'
  [ "${SKIP_UNNAMED:-0}" = 1 ] || echo "$all" | grep -E '__unnamed$'
}

model_complete() {                       # every condition of this model finished and clean for the split? (files are the source of truth)
  local c; for c in $(condition_order); do
    $PY tools/check_run.py --model "$1" --condition "$c" --split "$2" --max-parse-fail "$MAX_PARSE_FAIL" --quiet >/dev/null 2>&1 || return 1
  done
}

start_server() {                         # step 4: load the model and wait until it answers
  local alias=$1 hf=$2 devs tp patch extra="" waited=0
  IFS='|' read -r devs tp patch <<< "${GPUS[$alias]:-}"; [ -n "${devs:-}" ] || { log "no GPU config for $alias in GPUS"; return 1; }
  if [ "$patch" = 1 ]; then
    mkdir -p templates; $PY tools/make_olmo_template.py "$hf" "templates/$alias.jinja" >> "$LOGDIR/$alias.template.log" 2>&1 \
      || { log "chat-template patch failed for $alias (see $LOGDIR/$alias.template.log)"; return 1; }
    extra="--chat-template templates/$alias.jinja"
  fi
  log "loading $hf on GPUs $devs (tensor-parallel $tp); first load downloads the weights"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$devs VLLM_USE_FLASHINFER_SAMPLER=0 nohup $VLLM_BIN serve "$hf" --port "$PORT" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" --tensor-parallel-size "$tp" --generation-config vllm $extra > "$LOGDIR/$alias.vllm.log" 2>&1 &
  local pid=$!
  until [ "$(served_model)" = "$hf" ]; do
    kill -0 "$pid" 2>/dev/null || { log "vLLM exited early; last log lines:"; tail -n 12 "$LOGDIR/$alias.vllm.log" | tee -a "$LOGDIR/overnight.log"; return 1; }
    [ $waited -ge $((WAIT_MIN*60)) ] && { log "vLLM not ready after $WAIT_MIN min"; return 1; }
    sleep "$POLL"; waited=$((waited+POLL))
  done
  log "$hf is ready"
}

run_model_split() {                      # step 5: every condition, two attempts each; error rows are purged between attempts
  local alias=$1 split=$2 c attempt rc failed=0
  for c in $(condition_order); do
    rc=1
    for attempt in 1 2; do
      $PY run_condition.py --model "$alias" --condition "$c" --split "$split" --workers "$WORKERS" --base-url "http://localhost:$PORT/v1" >> "$LOGDIR/$alias.$split.run.log" 2>&1
      $PY tools/check_run.py --model "$alias" --condition "$c" --split "$split" --purge-errors --max-parse-fail "$MAX_PARSE_FAIL" > "$LOGDIR/.chk" 2>&1; rc=$?
      tee -a "$LOGDIR/overnight.log" < "$LOGDIR/.chk" > /dev/null; tail -n 1 "$LOGDIR/.chk"
      [ $rc -eq 0 ] || [ $rc -eq 2 ] && break                    # 2 = unparsable outputs: a retry cannot fix that
    done
    printf '%s\t%s\t%s\t%s\n' "$alias" "$split" "$c" "$([ $rc -eq 0 ] && echo ok || echo FAIL)" >> "$STATUS"
    [ $rc -eq 0 ] || failed=1
  done
  return $failed
}

phase() {
  local split=$1 alias hf t0
  for alias in $MODELS; do
    if [ "$split" = test ] && [ "$GATE" = lenient ] && ! model_complete "$alias" dev; then log "$alias: dev not clean, skipping test (GATE=lenient)"; continue; fi
    if model_complete "$alias" "$split"; then log "$alias/$split already complete, skipping"; continue; fi
    hf=$(hf_id "$alias"); t0=$SECONDS; CURRENT_HF=$hf
    unload_all
    if ! start_server "$alias" "$hf"; then printf '%s\t%s\t(load)\tFAIL\n' "$alias" "$split" >> "$STATUS"; unload_all; delete_cache "$hf"; CURRENT_HF=""; continue; fi
    log "=== $alias / $split: running ==="
    run_model_split "$alias" "$split" || log "$alias/$split: some conditions FAILED"
    log "=== $alias / $split finished in $(( (SECONDS-t0)/60 )) min ==="
    unload_all; delete_cache "$hf"; CURRENT_HF=""
  done
}

guard() { pgrep -u "$ME" -f "run_condition.py" > /dev/null && { echo "run_condition.py is still running (another generation loop?). Stop it first, or this script would unload its server."; exit 1; }; return 0; }

guard
log "start: models [$MODELS], phases [$PHASES], gate $GATE, logs in $LOGDIR"
unload_all                                                          # steps 1+2 before anything else
for split in $PHASES; do
  if [ "$split" = test ] && [ "$GATE" = strict ]; then
    for alias in $MODELS; do
      model_complete "$alias" dev || { log "TEST PHASE NOT STARTED: dev is not clean for $alias (GATE=strict). See $STATUS"; split=""; break; }
    done
    [ -n "$split" ] || break
  fi
  phase "$split"
done
log "summary: $(grep -c 'ok$' "$STATUS") ok, $(grep -c 'FAIL$' "$STATUS") FAIL (details: $STATUS)"
grep 'FAIL$' "$STATUS" || true
[ "$(grep -c 'FAIL$' "$STATUS")" -eq 0 ]