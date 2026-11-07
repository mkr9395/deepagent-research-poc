"""Offline smoke test of the orchestrator: scripted models + fake Tavily, no network/keys."""
from deepagents.backends import FilesystemBackend

from fin_agent.agents.orchestrator import build_orchestrator, validate_report
from fin_agent.research_guard import REPORT_MAX_WORDS
from fin_agent.tools import tavily_search as ts
from fin_agent.tools.tavily_search import Limits, TavilyGuard
from langchain_core.messages import AIMessage
from tests.test_researcher import GOOD_NOTE, ScriptedModel, call

PATH = "/notes/msft-googl-revenue.md"
REPORT = ("# MSFT vs GOOGL\n## Summary\n- Revenue up (https://sec.gov/a)\n"
          "## Coverage and Gaps\n- Revenue: covered\n## Sources\nhttps://sec.gov/a\n")


def test_validate_report():
    assert validate_report(REPORT) is None
    assert "limit is 700" in validate_report(REPORT + "w " * REPORT_MAX_WORDS)
    assert "Coverage" in validate_report("# T\nhttps://x.com")
    assert "source URLs" in validate_report("# T\n## Coverage and Gaps\nok")


def test_orchestrator_plans_delegates_reads_notes_and_saves_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=lambda q: {
        "query": q, "results": [{"title": "10-K", "url": "https://sec.gov/a", "content": "Revenue up"}]}))
    researcher = ScriptedModel(script=[
        call("tavily_search", query="q", task_id="t1"),
        call("think_tool", evidence_summary="ok", evidence_sufficient=True),
        call("write_file", file_path=PATH, content=GOOD_NOTE),
        AIMessage(content="- Revenue up"),
    ])
    out_file = tmp_path / "final_report.md"
    orch = ScriptedModel(script=[
        call("write_todos", todos=[{"content": "research", "status": "in_progress"}]),
        call("task", description="Revenue for MSFT and GOOGL", subagent_type="researcher"),
        call("write_file", file_path="/x.md", content="nope"),          # blocked
        call("read_file", file_path=PATH),
        call("save_final_report", markdown="# too short, no coverage"),  # rejected
        call("save_final_report", markdown=REPORT),
        AIMessage(content="done"),
    ])
    research_dir = tmp_path / "research"
    (research_dir / "notes").mkdir(parents=True)
    agent = build_orchestrator(
        model=orch, researcher_model=researcher, report_path=out_file,
        backend=FilesystemBackend(root_dir=research_dir, virtual_mode=True),
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "compare MSFT and GOOGL"}]})

    assert orch.idx == len(orch.script) and researcher.idx == len(researcher.script)
    assert (research_dir / "notes" / "msft-googl-revenue.md").is_file()   # note in sandbox
    assert out_file.read_text(encoding="utf-8").strip() == REPORT.strip()  # report outside it
    assert not (research_dir / "x.md").exists()                            # orchestrator can't write
    by_name = {m.name: m.content for m in out["messages"] if getattr(m, "name", None)}
    assert "write_todos" in by_name and by_name["task"].startswith(f"Notes file: {PATH}")
    assert "Saved final report" in by_name["save_final_report"]
