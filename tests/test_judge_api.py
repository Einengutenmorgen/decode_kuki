"""Judge API path against a fake OpenRouter-style server (no network, no real model).
Checks: extra-body fields and token budget reach the request, provenance is recorded, truncated replies become judge_error.
usage (repo root): python tests/test_judge_api.py"""
import itertools, json, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import config, judge

seen, n = [], itertools.count()
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); seen.append(body); i = next(n)
        truncated = "бояре плохие" in body["messages"][1]["content"]       # simulate reasoning eating the token budget
        text = "" if truncated else json.dumps({"reason": "fake", "verdict": judge.VERDICTS[i % 3]})
        out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "openai/gpt-6-luna-20260922", "provider": "OpenAI",
                          "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop", "message": {"role": "assistant", "content": text}}],
                          "usage": {"prompt_tokens": 600, "completion_tokens": 40, "total_tokens": 640}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 8767), H); threading.Thread(target=srv.serve_forever, daemon=True).start()

refs = judge.load_refs(); dev = [c for c in refs.values() if c["split"] == "dev"]
res = ROOT / "apitest.jsonl"
res.write_text("\n".join(json.dumps({"case_id": c["case_id"], "condition_id": "ctx-test__named", "model": "m", "parse_ok": True,
                                      "raw": json.dumps({"literal": "", "insider": c["insider"], "why": ""})}, ensure_ascii=False) for c in dev), encoding="utf-8")
out = config.JUDGE_DIR / "apitest__openai_gpt-6-luna-20260922__j2.jsonl"
try:
    extra = '{"reasoning": {"effort": "none"}, "provider": {"order": ["openai"], "allow_fallbacks": false}}'
    r = subprocess.run([sys.executable, "judge.py", "run", "--results", str(res), "--split", "dev", "--judge-model", "openai/gpt-6-luna-20260922",
                        "--base-url", "http://127.0.0.1:8767/v1", "--max-tokens", "600", "--extra-body", extra, "--workers", "8"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-400:]
    assert all(b["temperature"] == 0 and b["max_tokens"] == 600 and b["reasoning"] == {"effort": "none"} and b["provider"]["order"] == ["openai"]
               and "response_format" in b for b in seen); print(f"ok  all {len(seen)} requests carry max_tokens, reasoning, provider and the schema")
    rows = [json.loads(l) for l in open(out)]; good = [x for x in rows if x["verdict"] in judge.VERDICTS]
    assert good and all(x["served_model"] == "openai/gpt-6-luna-20260922" and x["served_provider"] == "OpenAI" and x["tokens_in"] == 600 for x in good)
    print(f"ok  {len(good)} judgments record served model, provider and token counts")
    err = [x for x in rows if x["verdict"] == "judge_error"]
    assert len(err) == 1 and "finish_reason=length" in err[0]["reason"]; print("ok  truncated reply -> judge_error with finish_reason in the message (3 attempts)")
    print("summary line:", [l for l in r.stdout.splitlines() if l.startswith("answered by")][0])
finally:
    for p in (res, out, Path(str(out) + ".lock")): p.unlink(missing_ok=True)
