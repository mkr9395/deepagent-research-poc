"""Demo UI for the Financial Company Comparison Agent (Streamlit, POC).

    streamlit run streamlit_app.py

Two modes:
  * Live run   - runs the real agent and shows planning, the parallel researchers, their
                 searches, the Tavily budget, the notes read and the verification as they happen.
  * Replay     - replays a saved run (outputs/ui_runs/*.json) with the same live-looking
                 progress and NO API calls or cost. Every live run is saved automatically.

The UI watches the agent through its existing log records (no agent code is changed).

Secrets: API keys are only read from the environment by the agent, exactly as the CLI does.
The UI never displays them (the sidebar shows set/missing only) and every piece of text that
reaches the page or a saved run file is passed through `redact()` first.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # the agent uses paths relative to the project root
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

RUNS_DIR = ROOT / "outputs" / "ui_runs"
KEY_VARS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY")
EXAMPLES = {
    "Microsoft vs Alphabet: growth, risks, outlook": (
        "Compare Microsoft and Alphabet on revenue growth and profitability from their latest annual "
        "earnings, regulatory and antitrust risks from their latest 10-K risk factors, and business "
        "outlook from management guidance. Research these as three separate research tasks."
    ),
    "Amazon vs Walmart: financials only": (
        "Compare Amazon and Walmart on revenue growth and profitability for the latest fiscal year."
    ),
}


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------
_KEY_SHAPES = re.compile(r"(sk-[A-Za-z0-9_\-]{20,}|tvly-[A-Za-z0-9_\-]{10,}|lsv2_[A-Za-z0-9_\-]{10,})")


def redact(text: str) -> str:
    """Remove API key values (exact env values and well-known key shapes) from any text."""
    text = str(text)
    for name in KEY_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    return _KEY_SHAPES.sub("[REDACTED]", text)


# ---------------------------------------------------------------------------
# Log records -> UI events
# ---------------------------------------------------------------------------
_TASK_NEW = re.compile(r"Research task (\d+) \((.+)\)\s*$", re.S)
_SEARCH = re.compile(
    r"Tavily call (\d+)/(\d+) \(task (\d+): (\d+)/(\d+)\)(?: \| gap: (.*?))? \| query=(.*)$", re.S
)
_START = re.compile(r"PARENT -> RESEARCHER \| brief (\d+) words.*?\| (.*)$", re.S)
_DONE = re.compile(
    r"RESEARCHER -> PARENT \| reply (\d+) words \| note=(\S+)(?: \((\d+) words.*?\))? \| kept out of parent: "
    r"(\d+) messages, (\d+) tool results \((\d+) searches\), (\d+) chars", re.S
)
_READ = re.compile(r"READ ARTIFACT\s+(\S+) \| (\d+) words")
_PLAN = re.compile(r"PLAN \| (\d+) task\(s\) for (.*?) \| skipped=(.*)$", re.S)


def _clean(text: str) -> str:
    return " ".join(text.strip().strip("'\"").split())


def parse(name: str, msg: str, t: float) -> dict | None:
    """Turn one agent log record into a UI event (or None for records the UI ignores)."""
    if name == "fin_agent.orchestrator" and (m := _PLAN.match(msg)):
        return {"k": "plan", "t": t, "n": int(m.group(1)), "dims": _clean(m.group(2)), "skipped": _clean(m.group(3))}
    if name == "fin_agent.tavily":
        if m := _TASK_NEW.match(msg):
            return {"k": "task_new", "t": t, "n": int(m.group(1)), "id": m.group(2).strip()}
        if m := _SEARCH.match(msg):
            return {"k": "search", "t": t, "call": int(m.group(1)), "budget": int(m.group(2)),
                    "task": int(m.group(3)), "of": int(m.group(4)), "gap": _clean(m.group(6) or ""),
                    "query": _clean(m.group(7))}
        return None
    if name == "fin_agent.context":
        if m := _START.match(msg):
            return {"k": "task_start", "t": t, "words": int(m.group(1)), "brief": _clean(m.group(2))}
        if m := _DONE.match(msg):
            return {"k": "task_done", "t": t, "reply_words": int(m.group(1)), "note": m.group(2),
                    "note_words": int(m.group(3) or 0), "msgs": int(m.group(4)), "tools": int(m.group(5)),
                    "searches": int(m.group(6)), "chars": int(m.group(7))}
        return None
    if name == "fin_agent.synthesis":
        if m := _READ.search(msg):
            return {"k": "read", "t": t, "path": m.group(1), "words": int(m.group(2))}
        if "final report saved" in msg:
            return {"k": "saved", "t": t}
        if msg.startswith("VERIFY"):
            return {"k": "verify", "t": t, "ok": "no obvious gaps" in msg, "text": _clean(msg.split("| ")[-1])}
        if msg.startswith("TIME LIMIT"):
            return {"k": "timeout", "t": t, "text": _clean(msg)}
    return None


def build_state(events: list[dict]) -> dict:
    s = {"n_tasks": 0, "dims": "", "tasks": {}, "starts": 0, "done": 0, "calls": 0, "budget": 15,
         "reads": [], "saved": False, "verify": None, "timeouts": []}
    for e in events:
        k = e["k"]
        if k == "plan":
            s["n_tasks"], s["dims"] = e["n"], e["dims"]
        elif k == "task_new":
            s["tasks"].setdefault(e["n"], {"id": e["id"], "searches": [], "done": None})
            s["tasks"][e["n"]]["id"] = e["id"]
        elif k == "search":
            t = s["tasks"].setdefault(e["task"], {"id": f"task-{e['task']}", "searches": [], "done": None})
            t["searches"].append(e)
            s["calls"], s["budget"] = max(s["calls"], e["call"]), e["budget"]
        elif k == "task_start":
            s["starts"] += 1
        elif k == "task_done":
            s["done"] += 1
            stem = Path(e["note"]).stem
            target = next((n for n, t in s["tasks"].items() if t["id"] == stem and not t["done"]), None)
            if target is None:
                target = next((n for n, t in sorted(s["tasks"].items()) if not t["done"]), None)
            if target is None:
                target = max(s["tasks"], default=0) + 1
                s["tasks"][target] = {"id": stem, "searches": [], "done": None}
            s["tasks"][target]["done"] = e
        elif k == "read":
            s["reads"].append(e)
        elif k == "saved":
            s["saved"] = True
        elif k == "verify":
            s["verify"] = e
        elif k == "timeout":
            s["timeouts"].append(e)
    s["n_tasks"] = max(s["n_tasks"], len(s["tasks"]))
    return s


def phase_of(s: dict, running: bool) -> tuple[str, float]:
    if s["verify"]:
        return "Done: report saved and verified" if s["verify"]["ok"] else "Done: report saved, gaps flagged", 1.0
    if s["saved"]:
        return "Verifying the report against the request", 0.92
    if s["reads"]:
        return "Synthesizing the comparison from the notes", 0.8
    if s["starts"]:
        total = max(s["n_tasks"], s["starts"])
        frac = s["done"] / total
        if s["done"] >= total:
            return "Research finished; the lead analyst is reading the notes", 0.72
        return f"Researching: {s['done']}/{total} researchers finished", 0.12 + 0.55 * frac
    if s["n_tasks"]:
        return "Plan accepted; delegating to researchers", 0.1
    return ("Planning the research..." if running else "Waiting"), 0.03 if running else 0.0


# ---------------------------------------------------------------------------
# Orchestrator steps (captured with LangChain callbacks; no agent code is touched)
# ---------------------------------------------------------------------------
STEP_LABELS = {
    "llm": ("🧠", "Reasoning (LLM call)"),
    "submit_research_plan": ("📝", "Submit research plan"),
    "write_todos": ("✅", "Update todo list"),
    "task": ("🔎", "Delegate to researcher"),
    "read_file": ("📖", "Read research note"),
    "save_final_report": ("💾", "Save final report"),
    "verify_final_report": ("🔍", "Verify report"),
}


def md_safe(text: str) -> str:
    """Streamlit renders $...$ as math; dollar figures in financial text must be escaped."""
    return str(text).replace("$", "\\$")


def _short(text, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _result_text(output) -> str:
    content = getattr(output, "content", None)
    if content is None:  # a langgraph Command (the `task` tool returns one)
        update = getattr(output, "update", None)
        msgs = (update or {}).get("messages") if isinstance(update, dict) else None
        content = getattr(msgs[-1], "content", "") if msgs else output
    return content if isinstance(content, str) else str(content)


def describe_step(tool: str, args: dict) -> str:
    if tool == "llm":
        return "deciding the next action"
    if tool == "submit_research_plan":
        tasks = args.get("tasks") or []
        qs = "; ".join(_short(t.get("question", "") if isinstance(t, dict) else getattr(t, "question", ""), 90) for t in tasks)
        return _short(f"{len(tasks)} task(s): {qs}", 260)
    if tool == "write_todos":
        todos = args.get("todos") or []
        return _short(" · ".join(f"[{t.get('status', '?')}] {t.get('content', '')}" for t in todos if isinstance(t, dict)), 260)
    if tool == "task":
        return _short(args.get("description", ""), 220)
    if tool == "read_file":
        return str(args.get("file_path", ""))
    if tool == "save_final_report":
        return f"{len(str(args.get('markdown', '')).split())} words"
    if tool == "verify_final_report":
        return "re-read the saved report and check it against the original request"
    return _short(json.dumps(args, default=str), 160)


def describe_result(tool: str, text: str) -> str:
    if tool == "read_file":
        return f"loaded {len(text.split())} words into the lead's context"
    if tool == "task":
        return _short(text, 200)
    return _short(text, 220)


class StepTracker(BaseCallbackHandler):
    """Emits a step event for every tool call / LLM call made by the ORCHESTRATOR.

    Runs nested inside a `task` call (the researcher's own searches and LLM calls) are
    recognised through the callback parent chain and skipped; the researcher panels cover them."""

    raise_error = False

    def __init__(self, emit):
        self.emit = emit
        self._lock = threading.Lock()
        self._parent: dict = {}
        self._task_runs: set = set()
        self._shown: dict = {}  # run_id -> tool name, for steps that were announced

    def _inside_task(self, run_id, parent_run_id) -> bool:
        with self._lock:
            self._parent[run_id] = parent_run_id
            cur = parent_run_id
            while cur is not None:
                if cur in self._task_runs:
                    return True
                cur = self._parent.get(cur)
        return False

    def _start(self, run_id, tool: str, detail: str) -> None:
        self._shown[run_id] = tool
        self.emit({"k": "step", "id": str(run_id), "tool": tool, "detail": detail})

    def _end(self, run_id, result: str, error: bool = False) -> None:
        tool = self._shown.pop(run_id, None)
        if tool is not None:
            self.emit({"k": "step_end", "id": str(run_id), "result": result, "error": error})

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        inside = self._inside_task(run_id, parent_run_id)
        name = kwargs.get("name") or (serialized or {}).get("name") or ""
        if "Middleware" in name:  # e.g. "TodoListMiddleware.after_model"; names vary by deepagents version
            self.emit({"k": "mw", "name": name, "who": "researcher" if inside else "orchestrator"})

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        if not self._inside_task(run_id, parent_run_id):
            self._start(run_id, "llm", describe_step("llm", {}))

    def on_llm_end(self, response, *, run_id, **kwargs):
        orchestrator_call = run_id in self._shown
        self._end(run_id, "")
        if not orchestrator_call:
            return
        # `write_todos` runs without a tool callback in this framework, so read it from the model's own message.
        try:
            calls = getattr(response.generations[0][0].message, "tool_calls", None) or []
        except (IndexError, AttributeError):
            calls = []
        for tc in calls:
            if tc.get("name") == "write_todos":
                sid = str(tc.get("id") or f"todos-{run_id}")
                todos = [{"content": str(t.get("content", "")), "status": str(t.get("status", "pending"))}
                         for t in (tc.get("args") or {}).get("todos") or [] if isinstance(t, dict)]
                self.emit({"k": "step", "id": sid, "tool": "write_todos", "todos": todos,
                           "detail": describe_step("write_todos", tc.get("args") or {})})
                self.emit({"k": "step_end", "id": sid, "result": "todo list updated", "error": False})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, f"{type(error).__name__}", error=True)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        tool = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        if tool == "task":
            with self._lock:
                self._task_runs.add(run_id)
        if self._inside_task(run_id, parent_run_id):
            return
        args = inputs if isinstance(inputs, dict) else {"input": input_str}
        self._start(run_id, tool, describe_step(tool, args))

    def on_tool_end(self, output, *, run_id, **kwargs):
        tool = self._shown.get(run_id, "")
        self._end(run_id, describe_result(tool, _result_text(output)))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, _short(f"{type(error).__name__}: {error}", 200), error=True)


def derive_step_events(events: list[dict]) -> list[dict]:
    """For runs saved before step capture existed: rebuild a coarse step list from the log events."""
    out, open_tasks = [], []

    def add(i, tool, detail, t, result="", error=False):
        out.append({"k": "step", "t": t, "id": i, "tool": tool, "detail": detail})
        if result is not None:
            out.append({"k": "step_end", "t": t, "id": i, "result": result, "error": error})

    for n, e in enumerate(events):
        k = e["k"]
        if k == "plan":
            add(f"d{n}", "submit_research_plan", f"{e['n']} task(s) for {e['dims']}", e["t"], "Plan accepted")
        elif k == "task_start":
            sid = f"d{n}"
            add(sid, "task", e["brief"], e["t"], None)
            open_tasks.append(sid)
        elif k == "task_done" and open_tasks:
            out.append({"k": "step_end", "t": e["t"], "id": open_tasks.pop(0), "result": f"Notes file: {e['note']} ({e['reply_words']} words returned)", "error": False})
        elif k == "read":
            add(f"d{n}", "read_file", e["path"], e["t"], f"loaded {e['words']} words into the lead's context")
        elif k == "saved":
            add(f"d{n}", "save_final_report", "final report", e["t"], "Saved final report")
        elif k == "verify":
            add(f"d{n}", "verify_final_report", "check report vs request", e["t"], "No obvious gaps" if e["ok"] else e["text"])
    return out


def collect_steps(events: list[dict]) -> tuple[list[dict], bool]:
    derived = not any(e["k"] == "step" for e in events)
    src = derive_step_events(events) if derived else events
    steps, idx = [], {}
    for e in src:
        if e["k"] == "step":
            idx[e["id"]] = len(steps)
            steps.append({**e, "end": None})
        elif e["k"] == "step_end" and e["id"] in idx:
            steps[idx[e["id"]]]["end"] = e
    return steps, derived


def render_todos(steps: list[dict]) -> None:
    todos = next((s["todos"] for s in reversed(steps) if s.get("todos")), None)
    if not todos:
        return
    icons = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}
    done = sum(t["status"] == "completed" for t in todos)
    st.markdown(f"**Plan** ({done}/{len(todos)} done)")
    for t in todos:
        st.markdown(f"{icons.get(t['status'], '⬜')} {md_safe(t['content'])}")


def render_middleware(events: list[dict]) -> None:
    counts: dict = {}
    for e in events:
        if e["k"] == "mw":
            key = (e["who"], e["name"])
            counts[key] = counts.get(key, 0) + 1
    if counts:
        with st.expander(f"Middleware activity ({sum(counts.values())} hook calls)", expanded=False):
            for (who, name), n in sorted(counts.items()):
                st.caption(f"{who} · `{name}` ×{n}")


def render_steps(events: list[dict], running: bool) -> None:
    steps, derived = collect_steps(events)
    if not steps:
        return
    render_todos(steps)
    render_middleware(events)
    with st.expander(f"Orchestrator steps ({len(steps)})", expanded=True):
        if derived:
            st.caption("Reconstructed from the run log (this run was saved before step capture existed).")
        for i, step in enumerate(steps, start=1):
            end = step["end"]
            icon, label = STEP_LABELS.get(step["tool"], ("🔧", step["tool"]))
            mark = "❌" if end and end.get("error") else ("⏳" if end is None and running else "✔️")
            st.markdown(f"{i}. {mark} {icon} **{label}** · {md_safe(step['detail'])} &nbsp; `+{step['t']:.0f}s`")
            if end and end.get("result"):
                st.caption(f"↳ {md_safe(end['result'])}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(ph, events: list[dict], elapsed: float, running: bool) -> None:
    s = build_state(events)
    phase, frac = phase_of(s, running)
    with ph.container():
        c = st.columns(4)
        c[0].metric("Elapsed", f"{elapsed:.0f}s")
        c[1].metric("Tavily searches", f"{s['calls']}/{s['budget']}")
        c[2].metric("Researchers done", f"{s['done']}/{max(s['n_tasks'], s['starts'], 0)}")
        c[3].metric("Notes read by lead", f"{len(s['reads'])}/{s['done']}" if s["done"] else "0")
        st.progress(min(frac, 1.0), text=phase)
        if s["dims"]:
            st.caption(f"Requested dimensions: {s['dims']}")
        for tmo in s["timeouts"]:
            st.warning(tmo["text"])

        render_steps(events, running)

        n = max(s["n_tasks"], 1) if (s["n_tasks"] or s["tasks"]) else 0
        if n:
            cols = st.columns(n)
            for i, col in enumerate(cols, start=1):
                task = s["tasks"].get(i)
                with col, st.container(border=True):
                    if not task:
                        st.markdown(f"**Researcher {i}**")
                        st.caption("starting...")
                        continue
                    status = "done" if task["done"] else "searching..."
                    st.markdown(f"**Researcher {i}** · `{task['id']}`")
                    st.caption(f"status: {status} · {len(task['searches'])} search(es)")
                    for srch in task["searches"]:
                        st.markdown(f"- `#{srch['call']}` {md_safe(srch['query'])}")
                        if srch["gap"]:
                            st.caption(f"  gap: {md_safe(srch['gap'])}")
                    d = task["done"]
                    if d:
                        st.success(
                            f"Note saved: `{Path(d['note']).name}` ({d['note_words']} words)\n\n"
                            f"Parent received **{d['reply_words']} words**. Kept out of the parent's "
                            f"context: {d['msgs']} messages, {d['tools']} tool results, "
                            f"{d['chars']:,} chars."
                        )
        if s["reads"]:
            st.caption("Lead analyst read: " + ", ".join(f"`{Path(r['path']).name}` ({r['words']}w)" for r in s["reads"]))
        if s["verify"]:
            (st.success if s["verify"]["ok"] else st.warning)(
                "Verification: " + ("no obvious gaps" if s["verify"]["ok"] else s["verify"]["text"])
            )


def render_results(rec: dict) -> None:
    st.divider()
    st.subheader("Result")
    if rec.get("status") == "timeout":
        st.error("The overall run time limit was exceeded; no final report was saved.")
    elif rec.get("status") == "error":
        st.error(f"The run failed: {rec.get('error', 'unknown error')}")
    tab_report, tab_notes, tab_usage, tab_log = st.tabs(["Final report", "Research notes", "LLM usage", "Activity log"])
    with tab_report:
        if rec.get("report_md"):
            st.markdown(md_safe(rec["report_md"]))
            st.download_button("Download report (.md)", rec["report_md"], file_name=Path(rec.get("report_path") or "report.md").name,
                               mime="text/markdown")
        else:
            st.info("No final report was saved for this run.")
    with tab_notes:
        st.caption("Detailed findings live on disk; the lead analyst keeps only short summaries and reads these at synthesis time.")
        for name, text in (rec.get("notes") or {}).items():
            with st.expander(f"{name} ({len(text.split())} words)"):
                st.markdown(md_safe(text))
        if not rec.get("notes"):
            st.info("No notes were produced.")
    with tab_usage:
        st.code(rec.get("usage") or "No usage recorded.", language="text")
    with tab_log:
        for e in rec.get("events", []):
            st.text(f"+{e['t']:6.1f}s  {e['k']:<10} " + redact(json.dumps({k: v for k, v in e.items() if k not in ('k', 't')})[:240]))


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------
class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put((record.name, redact(record.getMessage()), time.time()))
        except Exception:  # noqa: BLE001 - logging must never break the agent
            pass


@st.cache_resource
def run_lock() -> threading.Lock:
    return threading.Lock()  # one live run at a time (the Tavily guard and notes folder are shared)


def run_live(query: str, ph) -> dict:
    import main as cli  # the CLI module: reuses its setup helpers, nothing is modified
    from fin_agent.agents.orchestrator import build_orchestrator
    from fin_agent.agents.researcher import RESEARCH_DIR, get_backend
    from fin_agent.request import save_original_request
    from fin_agent.tools.tavily_search import start_request
    from fin_agent.usage import UsageTracker

    load_dotenv()
    save_original_request(query)
    get_backend()
    cli.reset_run_artifacts()
    start_request()
    agent, usage = build_orchestrator(), UsageTracker()

    q: queue.Queue = queue.Queue()
    steps = StepTracker(lambda p: q.put(("__step__", {k: redact(v) if isinstance(v, str) else v for k, v in p.items()}, time.time())))
    handler = _QueueHandler(q)
    agent_log = logging.getLogger("fin_agent")
    old_level = agent_log.level
    agent_log.setLevel(logging.INFO)
    agent_log.addHandler(handler)

    box: dict = {}

    def work() -> None:
        try:
            box["result"] = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 100, "run_name": "financial-comparison-ui", "tags": ["fin-agent-poc", "ui"],
                        "metadata": {"query": query[:300]}, "callbacks": [usage, steps]},
            )
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t0 = time.time()
    limit = cli.run_time_limit_seconds()
    events: list[dict] = []
    worker = threading.Thread(target=work, daemon=True, name="agent-run")
    worker.start()

    def drain() -> None:
        while True:
            try:
                name, msg, ts = q.get_nowait()
            except queue.Empty:
                return
            if name == "__step__":
                events.append({**msg, "t": ts - t0})
            elif (ev := parse(name, msg, ts - t0)) is not None:
                events.append(ev)

    try:
        while worker.is_alive() and (not limit or time.time() - t0 < limit):
            drain()
            render(ph, events, time.time() - t0, True)
            time.sleep(0.4)
        drain()
    finally:
        agent_log.removeHandler(handler)
        agent_log.setLevel(old_level)

    rec = {"query": query, "events": events, "duration": time.time() - t0, "status": "ok",
           "usage": redact(usage.summary()), "saved_at": datetime.now().isoformat(timespec="seconds")}
    if worker.is_alive():
        rec["status"] = "timeout"
    elif "error" in box:
        rec["status"], rec["error"] = "error", redact(f"{type(box['error']).__name__}: {box['error']}")
    else:
        saved = [m.text.split(" to ", 1)[1].rsplit(" (", 1)[0] for m in box["result"]["messages"]
                 if getattr(m, "name", None) == "save_final_report" and m.text.startswith("Saved final report to ")]
        if saved and Path(saved[-1]).is_file():
            rec["report_path"] = saved[-1]
            rec["report_md"] = redact(Path(saved[-1]).read_text(encoding="utf-8"))
    rec["notes"] = {p.name: redact(p.read_text(encoding="utf-8")) for p in sorted((RESEARCH_DIR / "notes").glob("*.md"))}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def saved_runs() -> list[Path]:
    return sorted(RUNS_DIR.glob("*.json"), reverse=True) if RUNS_DIR.is_dir() else []


def replay(rec: dict, ph, speed: float) -> None:
    events, shown, last_t = rec["events"], [], 0.0
    render(ph, shown, 0.0, True)
    for e in events:
        if speed > 0:
            time.sleep(min((e["t"] - last_t) / speed, 2.5))
        last_t = e["t"]
        shown.append(e)
        render(ph, shown, e["t"], True)
    render(ph, events, rec.get("duration", last_t), False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Financial Comparison Agent", page_icon="📊", layout="wide")
    load_dotenv()
    st.title("📊 Financial Company Comparison Agent")
    st.caption("LangChain Deep Agents POC: one lead analyst plans the work and delegates to parallel researcher "
               "subagents; findings live on disk and the lead keeps only short summaries.")

    with st.sidebar:
        st.header("Setup")
        for name in KEY_VARS[:2]:
            st.write(("✅" if os.environ.get(name) else "❌") + f" `{name}` " + ("set" if os.environ.get(name) else "missing"))
        st.caption("Keys are read from the environment only and are never displayed.")
        st.divider()
        mode = st.radio("Mode", ["Live run", "Replay saved run"], help="Replay uses no API calls and costs nothing.")
        st.caption("Live run: ~1-3 minutes, uses OpenAI + Tavily credit (max 15 searches).")

    ph = st.empty()
    rec = st.session_state.get("record")

    if mode == "Live run":
        choice = st.selectbox("Example request", ["(write your own)"] + list(EXAMPLES))
        query = st.text_area("Comparison request", value=EXAMPLES.get(choice, ""), height=110,
                             placeholder="Compare Microsoft and Alphabet on revenue growth, profitability, risks and outlook")
        have_keys = all(os.environ.get(k) for k in KEY_VARS[:2])
        if not have_keys:
            st.warning("OPENAI_API_KEY / TAVILY_API_KEY are not set in the environment, so a live run is unavailable. "
                       "Use Replay mode.")
        if st.button("Run comparison", type="primary", disabled=not (have_keys and query.strip())):
            lock = run_lock()
            if not lock.acquire(blocking=False):
                st.warning("Another run is in progress. Wait for it to finish.")
            else:
                try:
                    rec = st.session_state["record"] = run_live(query.strip(), ph)
                finally:
                    lock.release()
    else:
        runs = saved_runs()
        if not runs:
            st.info("No saved runs yet. Do a live run first; every live run is saved to outputs/ui_runs/.")
        else:
            def label(p: Path) -> str:
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    return f"{d.get('saved_at', p.stem)} - {d.get('query', '')[:70]}"
                except (OSError, ValueError):
                    return p.stem
            pick = st.selectbox("Saved run", runs, format_func=label)
            speed = {"Real time": 1.0, "4x faster": 4.0, "Instant": 0.0}[
                st.radio("Replay speed", ["Real time", "4x faster", "Instant"], index=1, horizontal=True)]
            if st.button("Replay", type="primary"):
                rec = st.session_state["record"] = json.loads(pick.read_text(encoding="utf-8"))
                st.session_state["replayed"] = True
                st.markdown(f"**Request:** {rec['query']}")
                replay(rec, ph, speed)

    if rec:
        render(ph, rec["events"], rec.get("duration", 0.0), False)  # final state survives reruns (e.g. downloads)
        render_results(rec)


main()
