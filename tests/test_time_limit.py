"""Offline smoke test of the per-task time limit: a researcher that is too slow is abandoned,
the orchestrator is told, and the final report must disclose it."""
import threading
import time

from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage

from fin_agent.agents.orchestrator import build_orchestrator
from fin_agent.agents import orchestrator as orch_mod
from fin_agent.research_guard import TaskClock
from tests.test_researcher import GOOD_NOTE, ScriptedModel, call

PATH = "/notes/slow-task.md"
OK_REPORT = ("# A vs B\n## Executive Summary\n- Risks not researched: the research task exceeded the "
             "time limit (https://x.com/a)\n## Evidence gaps\n- time limit exceeded\n## Sources\nhttps://x.com/a\n")


class SlowModel(ScriptedModel):
    delay: float = 1.5

    def _generate(self, *a, **kw):
        time.sleep(self.delay)
        return super()._generate(*a, **kw)


def test_clock_soft_and_hard_limits():
    t = [0.0]
    clock = TaskClock(100, now=lambda: t[0])
    assert not clock.soft_expired() and not clock.hard_expired()
    t[0] = 71
    assert clock.soft_expired() and not clock.hard_expired()
    t[0] = 100
    assert clock.hard_expired()


def test_slow_researcher_is_abandoned_and_report_must_disclose_it(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "REPORTS_DIR", tmp_path)
    research = tmp_path / "research"
    (research / "notes").mkdir(parents=True)
    request_file = tmp_path / "original_request.md"
    request_file.write_text("# Original Request\n\n## User question\n\ncompare A and B on risks\n", encoding="utf-8")

    slow = SlowModel(script=[call("write_file", file_path=PATH, content=GOOD_NOTE), AIMessage(content="- x")])
    orch = ScriptedModel(script=[
        call("submit_research_plan", companies=["A", "B"], dimensions=["risks"],
             tasks=[{"question": "Risks?", "dimensions": ["risks"], "why_needed": "asked"}]),
        call("task", description="Risks for A and B", subagent_type="researcher"),   # times out
        call("save_final_report", markdown="# A vs B\n## Summary\nhttps://x.com/a\n"),  # refused: no disclosure
        call("save_final_report", markdown=OK_REPORT),
        AIMessage(content="done"),
    ])
    import fin_agent.agents.orchestrator as o
    real = o.OrchestratorGuardMiddleware.__init__
    monkeypatch.setattr(o.OrchestratorGuardMiddleware, "__init__",
                        lambda self, planner, timeout=None: real(self, planner, timeout=0.5))

    agent = build_orchestrator(
        model=orch, researcher_model=slow, request_path=request_file,
        backend=FilesystemBackend(root_dir=research, virtual_mode=True),
    )
    t0 = time.monotonic()
    out = agent.invoke({"messages": [{"role": "user", "content": "compare A and B on risks"}]})
    elapsed = time.monotonic() - t0

    assert elapsed < 1.4, f"orchestrator waited for the slow researcher ({elapsed:.1f}s)"
    by = {m.name: m.content for m in out["messages"] if getattr(m, "name", None)}
    assert "TIME LIMIT EXCEEDED" in by["task"] and "NOT GENERATED" in by["task"]
    saves = [m.content for m in out["messages"] if getattr(m, "name", None) == "save_final_report"]
    assert "time limit" in saves[0] and "Nothing was saved" in saves[0]   # undisclosed report refused
    assert "Saved final report" in saves[1]                               # disclosed report saved
    assert "time limit" in (tmp_path / "a-b_final_report.md").read_text(encoding="utf-8")

    time.sleep(2.0)  # the abandoned researcher must wind down on its own, not keep working
    assert not any(t.name == "researcher-task" and t.is_alive() for t in threading.enumerate())
    assert not (research / "notes" / "slow-task.md").exists()  # cancelled before it could write
