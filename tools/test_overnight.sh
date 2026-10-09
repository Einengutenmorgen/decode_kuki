#!/usr/bin/env bash
# Test of overnight.sh with a fake vLLM (no GPU, no network). Runs in a temporary copy of the repo: real results and caches are untouched.
# usage: bash tools/test_overnight.sh [A|B]      A = a model is already loaded, dev then test (about 4 min); B = crash, failing requests, gate (about 2 min)
set -uo pipefail
if pgrep -u "${USER:-$(id -un)}" -f "vllm serve|VLLM::EngineCore|run_condition.py" > /dev/null; then
  echo "Refusing to run: a real vLLM server or run_condition.py of yours is running, and overnight.sh would stop it. Stop them first."; exit 2; fi
R=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d "$R/.test_tmp.XXXXXX"); W=$T/repo; mkdir -p "$W" "$T/bin"
cp "$R"/*.py "$R"/overnight.sh "$W"/ && cp -r "$R/tools" "$W/tools" && ln -s "$R/datasets" "$W/datasets"
cat > "$T/bin/vllm" <<'PYEOF'
#!/usr/bin/env python3
# fake `vllm serve <model> --port N`: writes a fake cache entry, serves /v1/models and /v1/chat/completions
# env: FAKE_CRASH_MODEL (exit on start), FAKE_FAIL_MODEL (answer every request with HTTP 400)
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
model, port = sys.argv[2], int(sys.argv[sys.argv.index("--port") + 1])
if os.environ.get("FAKE_CRASH_MODEL") == model: print("fake crash on start", flush=True); sys.exit(1)
cache = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + model.replace("/", "--")); os.makedirs(cache, exist_ok=True)
open(os.path.join(cache, "weights.safetensors"), "w").write("x" * 1000); time.sleep(1)
class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        out = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def do_GET(self): self._send(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if os.environ.get("FAKE_FAIL_MODEL") == model: return self._send(400, {"error": {"message": "fake failure", "type": "invalid_request_error"}})
        unnamed = "Find up to 5 coded expressions" in body["messages"][1]["content"]
        text = json.dumps({"items": []} if unnamed else {"literal": "x", "insider": "y", "why": "z"})
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": model,
                         "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]})
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PYEOF
chmod a-x "$T/bin/vllm" "$W/overnight.sh"; cd "$W"      # nothing is executed directly: works on a noexec /tmp too
export HF_HUB_CACHE=$T/hf PY=${PY:-python3} VLLM_BIN="${PY:-python3} $T/bin/vllm" POLL=1 PORT=8791 WORKERS=8 LOGDIR=$T/logs
mkdir -p "$HF_HUB_CACHE"; ok() { echo "ok  $*"; }; die() { echo "FAIL $*"; echo "(details kept in $T)"; exit 1; }; alive() { pgrep -f "$T/bin/vllm serve" > /dev/null; }

scenario_A() {
  echo "--- A: a model is already loaded; one model, dev then test ---"
  $VLLM_BIN serve allenai/Olmo-3-7B-Instruct --port $PORT > "$T/fake_setup.log" 2>&1 &
  for i in $(seq 30); do curl -s localhost:$PORT/v1/models > /dev/null && break; sleep 1; done
  [ -d "$HF_HUB_CACHE/models--allenai--Olmo-3-7B-Instruct" ] || die "setup: fake server did not start: $(tail -n 3 "$T/fake_setup.log" 2>&1)"
  MODELS="gemma4-e4b" bash ./overnight.sh > "$T/A.out" 2>&1; rc=$?
  [ $rc -eq 0 ] && ok "exit code 0" || { tail -20 "$T/A.out"; die "exit code $rc"; }
  grep -q "vLLM is running (model: allenai/Olmo-3-7B-Instruct)" "$T/A.out" && [ ! -d "$HF_HUB_CACHE/models--allenai--Olmo-3-7B-Instruct" ] && ok "running server stopped, its cache deleted" || die "unload"
  [ -z "$(ls "$HF_HUB_CACHE")" ] && ok "cache empty at the end" || die "cache: $(ls "$HF_HUB_CACHE")"
  alive && die "fake vllm still running" || ok "no vLLM process left"
  [ "$(grep -c 'ok$' "$LOGDIR/status.tsv")" -eq 20 ] && ok "20/20 condition runs ok (10 dev + 10 test)" || die "status: $(head "$LOGDIR/status.tsv")"
  [ "$(grep -n '=== gemma4-e4b / dev: running' "$T/A.out" | cut -d: -f1)" -lt "$(grep -n '=== gemma4-e4b / test: running' "$T/A.out" | cut -d: -f1)" ] && ok "dev ran before test" || die "order"
  [ "$(wc -l < results/ctx-paragraph__title-0__outlet-none__en__named__gemma4-e4b__v2.jsonl)" -eq 1046 ] && ok "1046 results (50 dev + 996 test)" || die "result rows"
  MODELS="gemma4-e4b" bash ./overnight.sh > "$T/A2.out" 2>&1; grep -q "already complete" "$T/A2.out" && ! grep -q "loading google/" "$T/A2.out" && ok "restart skips finished work, loads nothing" || die "restart"
}
scenario_B() {
  echo "--- B: crash on load, failing requests, strict gate ---"
  rm -rf results; export LOGDIR=$T/logsB FAKE_CRASH_MODEL=google/gemma-4-E4B-it FAKE_FAIL_MODEL=google/gemma-4-12B-it
  MODELS="gemma4-e4b gemma4-12b gemma4-31b" bash ./overnight.sh > "$T/B.out" 2>&1; rc=$?
  [ $rc -ne 0 ] && ok "non-zero exit code when something failed" || die "exit code 0"
  grep -q "vLLM exited early" "$T/B.out" && grep -q $'gemma4-e4b\tdev\t(load)\tFAIL' "$LOGDIR/status.tsv" && ok "crash on load detected and recorded, next model continued" || die "crash handling"
  [ "$(grep -c $'gemma4-12b\tdev\t.*\tFAIL' "$LOGDIR/status.tsv")" -eq 10 ] && ok "12B: 10/10 conditions FAIL after 2 attempts" || die "12b"
  [ "$(grep -c $'gemma4-31b\tdev\t.*\tok' "$LOGDIR/status.tsv")" -eq 10 ] && ok "31B: dev ok" || die "31b"
  grep -q "TEST PHASE NOT STARTED" "$T/B.out" && [ "$(grep -c $'\ttest\t' "$LOGDIR/status.tsv")" -eq 0 ] && ok "strict gate: no test run after dev failures" || die "gate"
  [ -z "$(ls "$HF_HUB_CACHE")" ] && ok "cache empty after failures too" || die "cache: $(ls "$HF_HUB_CACHE")"
  alive && die "fake vllm still running" || ok "no vLLM process left"
}
case "${1:-AB}" in *A*) scenario_A;; esac
case "${1:-AB}" in *B*) scenario_B;; esac
echo "overnight checks passed (${1:-AB})"; rm -rf "$T"