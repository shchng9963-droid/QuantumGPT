"""QuantumGPT LIVE — minimal SSE bridge between the monitor.html UI and a real
ReActAgent.run() invocation.

Why this exists
---------------
The default mission control replays pre-baked traces (web/static/missions.json
built from eval/results/...). It is deterministic, but it does NOT call the
LLM. This server wires the same UI to a real DeepSeek (or OpenAI) call so the
demo can actually exercise the agent end-to-end.

How it works
------------
- GET  /                  → serves web/static/monitor.html (with LIVE button injected)
- GET  /missions.json     → unchanged static asset (fallback for REPLAY buttons)
- GET  /static/<path>     → other static assets if any
- GET  /live/run?prompt=… → text/event-stream
                            spawns ReActAgent.run() in a background thread,
                            polls trace.steps every 200 ms, emits one SSE
                            event per newly-appeared TraceStep:
                              event: step  data: {"kind":"thought",...}
                              event: step  data: {"kind":"tool",...}
                              event: step  data: {"kind":"observe",...}
                            ends with:
                              event: claim     data: {"text": final_answer, ...}
                              event: verifier  data: {"verdict": "pass"|"fail", "reason": ...}
                              event: done      data: {"cost_usd":..., "tokens":..., "elapsed_s":...}

Run
---
    set -a; source .env; set +a
    .venv/bin/python web/live_server.py --port 8765

Then open http://127.0.0.1:8765/  and press the «≡ LIVE» button.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "web" / "static"
HTML_PATH = STATIC / "monitor.html"
MISSIONS_PATH = STATIC / "missions.json"


# ─────────────────────────────────────────────────────────────────────────────
# ReAct runner: invoke the real agent and stream steps as they appear.
# ─────────────────────────────────────────────────────────────────────────────
def _build_agent():
    """Lazy import so the server can boot even if heavy deps are missing —
    the LIVE endpoint will then return a friendly 500 instead of crashing.

    We override the OpenAI client with a tighter http timeout (25s) so a
    network-level stall surfaces as an exception inside agent.run() instead
    of hanging silently for the openai-python default of 600s.
    """
    from agent.react import ReActAgent
    from backends.fake_adapter import FakeBackendAdapter

    backend = FakeBackendAdapter("FakeBrisbane")
    agent = ReActAgent(
        backend=backend,
        provider="deepseek",
        verbose=False,
        use_mock=False,
        target_fidelity=0.85,
        max_tool_calls=12,
        max_seconds=60.0,
        use_memory=False,
    )
    # Tighten the LLM client http timeout. ReActAgent constructs an OpenAI
    # client with whatever the SDK default is (often 600s); replace it with
    # a 25s-per-request timeout so connection / read stalls fail fast and
    # agent.run() throws inside our try/except instead of hanging the SSE.
    try:
        from openai import OpenAI
        if agent.provider == "deepseek":
            agent.client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                timeout=25.0,
                max_retries=1,
            )
        elif agent.provider == "openai":
            kwargs = {
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "timeout": 25.0,
                "max_retries": 1,
            }
            base = os.environ.get("OPENAI_BASE_URL")
            if base:
                kwargs["base_url"] = base
            agent.client = OpenAI(**kwargs)
    except Exception:
        # If anything goes wrong, fall back to the default client.
        pass
    return agent


def _step_to_event(step) -> list[dict]:
    """One TraceStep can produce 1-3 SSE frames (thought / tool / observe)."""
    out: list[dict] = []
    i = step.step_num
    if step.thought:
        out.append({"i": i, "kind": "thought", "text": step.thought})
    if step.action:
        body = json.dumps(step.action_input or {}, ensure_ascii=False)
        out.append({
            "i": i, "kind": "tool",
            "tool": step.action,
            "text": f"{step.action}({body[:140]})",
        })
    if step.observation:
        ev: dict = {"i": i, "kind": "observe", "text": (step.observation or "")[:200]}
        if step.fidelity_observed is not None:
            ev["fidelity"] = float(step.fidelity_observed)
        out.append(ev)
    return out


def run_react_streaming(prompt: str, q: "queue.Queue[dict]") -> None:
    """Background thread: drive ReActAgent.run() and push events into q.
    Polls trace.steps every 0.2 s; emits new steps as they appear."""
    try:
        agent = _build_agent()
        q.put({"event": "boot", "data": {
            "provider": agent.provider, "model": agent.model,
            "backend": "FakeBrisbane",
        }})
    except Exception as e:
        q.put({"event": "error", "data": {
            "message": f"agent_init_failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }})
        q.put({"event": "done", "data": {}})
        return

    # Wrap run() so we can poll trace.steps from the outer poller thread.
    result_holder: dict = {}
    error_holder: dict = {}
    trace_holder: dict = {"trace": None}

    # Patch agent so we can grab the live trace before run() returns. The
    # cleanest way is to wrap _execute_loop or just reach in: ReActAgent
    # builds the trace at the top of run(). We monkey-patch instead by
    # snapshotting via a thread that calls run() and reads the trace from
    # the agent attributes the moment they're populated.
    def runner():
        try:
            trace = agent.run(prompt)
            trace_holder["trace"] = trace
            result_holder["final_answer"] = trace.final_answer
            result_holder["cost_usd"] = float(trace.cost_summary.get("total_cost_usd", 0.0))
            result_holder["prompt_tokens"] = int(trace.prompt_tokens_total or 0)
            result_holder["completion_tokens"] = int(trace.completion_tokens_total or 0)
            result_holder["elapsed_s"] = float(trace.elapsed_seconds or 0.0)
            result_holder["best_fidelity"] = float(trace.best_fidelity or 0.0)
            result_holder["num_tool_calls"] = int(trace.num_tool_calls or 0)
        except Exception as e:
            error_holder["err"] = f"{type(e).__name__}: {e}"
            error_holder["trace"] = traceback.format_exc(limit=6)

    th = threading.Thread(target=runner, daemon=True)
    th.start()

    # Poll for new trace steps. We need access to the trace object — the
    # ReActAgent appends to a local trace inside run(). To stream, we
    # patch by polling agent's last-known trace via a public attribute.
    # Workaround: ReActAgent doesn't expose live trace, so we patch its
    # _maybe_log_step or just track step count by attaching to agent.budget.
    # Simpler robust approach: poll budget.tool_calls + sleep until done,
    # then drain real trace at the end.
    last_emitted = 0
    started = time.time()

    while th.is_alive():
        # Try to peek into the agent's currently-running trace via attribute.
        live_trace = getattr(agent, "_live_trace_for_streaming", None) or trace_holder["trace"]
        if live_trace is not None:
            steps = list(live_trace.steps or [])
            while last_emitted < len(steps):
                for ev in _step_to_event(steps[last_emitted]):
                    q.put({"event": "step", "data": ev})
                last_emitted += 1
        # Heartbeat every 4 s so the SSE pipe and any proxy stay alive.
        if int(time.time() - started) % 4 == 0:
            q.put({"event": "ping", "data": {"elapsed": round(time.time() - started, 1)}})
        time.sleep(0.25)

    th.join(timeout=2)
    final_trace = trace_holder["trace"]
    if final_trace is not None:
        steps = list(final_trace.steps or [])
        while last_emitted < len(steps):
            for ev in _step_to_event(steps[last_emitted]):
                q.put({"event": "step", "data": ev})
            last_emitted += 1

    if error_holder:
        q.put({"event": "error", "data": {
            "message": error_holder["err"],
            "trace": error_holder.get("trace", ""),
        }})
        q.put({"event": "done", "data": {}})
        return

    final = result_holder.get("final_answer", "(no final answer)")
    q.put({"event": "claim", "data": {
        "text": final,
        "best_fidelity": result_holder.get("best_fidelity"),
    }})
    target = 0.85
    bf = result_holder.get("best_fidelity") or 0.0
    verdict = "pass" if bf >= target else "fail"
    reason = (
        f"best_fidelity={bf:.4f} ≥ target={target}"
        if verdict == "pass"
        else f"best_fidelity={bf:.4f} < target={target}"
    )
    q.put({"event": "verifier", "data": {"verdict": verdict, "reason": reason}})

    q.put({"event": "done", "data": {
        "cost_usd": result_holder.get("cost_usd", 0.0),
        "elapsed_s": result_holder.get("elapsed_s", 0.0),
        "tool_calls": result_holder.get("num_tool_calls", 0),
        "prompt_tokens": result_holder.get("prompt_tokens", 0),
        "completion_tokens": result_holder.get("completion_tokens", 0),
        "best_fidelity": result_holder.get("best_fidelity"),
    }})


# Patch ReActAgent so it exposes its in-flight trace under a known attribute.
# This is done at import time of the server, not the agent module, so it
# stays opt-in to the LIVE pathway.
def _install_live_trace_patch():
    try:
        from agent.react import ReActAgent
    except Exception:
        return
    orig_run = ReActAgent.run

    def patched_run(self, user_prompt: str, memory_context: str = ""):
        # We can't easily inject inside the original run without altering it.
        # Instead, we monkeypatch the trace creation to also stash a ref on self.
        # Simpler: rerun original but expose `_live_trace_for_streaming` after
        # it's available via an internal attribute the original run sets,
        # by overriding to capture the trace pointer when it appears.
        return orig_run(self, user_prompt, memory_context)

    # The hardest part is exposing the live trace mid-run. The original run()
    # builds `trace = AgentTrace(...)` and appends to `trace.steps`. We add a
    # tiny shim: after the trace is constructed, store it on self.
    # The cleanest hook is to subclass on the fly.
    import agent.react as _ar

    _orig_AgentTrace = _ar.AgentTrace

    class _StreamingAgentTrace(_orig_AgentTrace):
        pass

    # Replace AgentTrace globally so every instance is detectable; on
    # construction, the patched run will store the latest one on the agent.
    # However, AgentTrace is created inside run() with a bare class call,
    # so we need to intercept construction. Easiest: wrap __init__.
    _orig_init = _orig_AgentTrace.__init__

    _agent_ref: dict = {"agent": None}

    def _new_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        a = _agent_ref.get("agent")
        if a is not None:
            a._live_trace_for_streaming = self

    _orig_AgentTrace.__init__ = _new_init

    def patched_run2(self, user_prompt: str, memory_context: str = ""):
        _agent_ref["agent"] = self
        try:
            return orig_run(self, user_prompt, memory_context)
        finally:
            _agent_ref["agent"] = None

    ReActAgent.run = patched_run2


_install_live_trace_patch()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP / SSE handler
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "QuantumGPT-LIVE/0.1"

    def log_message(self, fmt, *args):
        # Suppress default access log noise; uncomment to debug.
        pass

    # ── static ──────────────────────────────────────────────────────────────
    def _send_static(self, path: Path, ctype: str):
        if not path.exists():
            self.send_error(404, "not found"); return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/monitor.html"):
            self._send_static(HTML_PATH, "text/html"); return
        if u.path == "/missions.json":
            self._send_static(MISSIONS_PATH, "application/json"); return
        if u.path.startswith("/static/"):
            p = STATIC / u.path[len("/static/"):]
            ctype = "text/plain"
            if p.suffix == ".css":  ctype = "text/css"
            elif p.suffix == ".js": ctype = "application/javascript"
            elif p.suffix == ".html": ctype = "text/html"
            self._send_static(p, ctype); return
        if u.path == "/live/run":
            qs = parse_qs(u.query)
            prompt = (qs.get("prompt", [""])[0] or "").strip()
            if not prompt:
                prompt = "transpile and run ghz_5 with ZNE; report mitigated fidelity"
            self._handle_live(prompt); return
        if u.path == "/live/narrate":
            qs = parse_qs(u.query)
            ctx = (qs.get("ctx", [""])[0] or "").strip()
            self._handle_narrate(ctx); return
        if u.path == "/healthz":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body); return
        self.send_error(404, "not found")

    def _handle_narrate(self, ctx: str):
        """One-shot LLM narration. Cheap (no tools, 60 token cap)."""
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"no api key"}')
            return
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url="https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None,
                timeout=15.0,
                max_retries=1,
            )
            sys_prompt = (
                "You are a quantum-backend monitoring AI. Given a JSON snapshot of the live "
                "FakeBrisbane backend, produce ONE short status note (≤25 words) in plain "
                "English. Match the tone of an on-call engineer's terse log. Do not add prefix "
                "labels. No bullets. No JSON."
            )
            t0 = time.time()
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": ctx or "{}"},
                ],
                max_tokens=60,
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            cost = 0.0
            if usage is not None:
                cost = (usage.prompt_tokens * 0.27 + usage.completion_tokens * 1.10) / 1_000_000
            payload = {
                "text": text,
                "elapsed_s": round(time.time() - t0, 3),
                "prompt_tokens": int(usage.prompt_tokens or 0) if usage else 0,
                "completion_tokens": int(usage.completion_tokens or 0) if usage else 0,
                "cost_usd": round(cost, 6),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            err = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    # ── SSE ─────────────────────────────────────────────────────────────────
    def _handle_live(self, prompt: str):
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            err = json.dumps({"message":
                "no DEEPSEEK_API_KEY / OPENAI_API_KEY in env — start server with: "
                "set -a; source .env; set +a; .venv/bin/python web/live_server.py"})
            self.wfile.write(b"event: error\ndata: " + err.encode() + b"\n\n")
            self.wfile.write(b"event: done\ndata: {}\n\n")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx/cloudflare buffering
        self.end_headers()

        q: queue.Queue[dict] = queue.Queue()
        worker = threading.Thread(
            target=run_react_streaming, args=(prompt, q), daemon=True
        )
        worker.start()

        def write_event(name: str, data: dict) -> bool:
            try:
                payload = json.dumps(data, ensure_ascii=False, default=str)
                self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        # Initial hello so the client knows the pipe is open
        if not write_event("hello", {"prompt": prompt}):
            return

        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                if not write_event("ping", {}):
                    return
                if not worker.is_alive() and q.empty():
                    return
                continue
            if not write_event(item.get("event", "step"), item.get("data", {})):
                return
            if item.get("event") == "done":
                return


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", type=str, default="0.0.0.0")
    args = ap.parse_args()

    print(f"[live_server] static dir: {STATIC}")
    print(f"[live_server] monitor:    http://127.0.0.1:{args.port}/")
    print(f"[live_server] live SSE:   http://127.0.0.1:{args.port}/live/run?prompt=...")
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("[live_server] ⚠ no API key — LIVE endpoint will return a friendly error.")
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
