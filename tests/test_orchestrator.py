"""Offline smoke test of the orchestrator: scripted models + fake Tavily, no network/keys."""
from deepagents.backends import FilesystemBackend

from fin_agent.agents.orchestrator import build_orchestrator, validate_report
from fin_agent.tools import tavily_search as ts
from fin_agent.tools.tavily_search import Limits, TavilyGuard
from langchain_core.messages import AIMessage
import pytest
from fin_agent.agents.orchestrator import ResearchPlanner, ResearchTask
from tests.test_researcher import GOOD_NOTE, ScriptedModel, call

PATH = "/notes/msft-googl-revenue.md"
REPORT = ("# MSFT vs GOOGL\n## Summary\n- Revenue up (https://sec.gov/a)\n"
          "## Coverage and Gaps\n- Revenue: covered\n## Sources\nhttps://sec.gov/a\n")


FIXED = REPORT + "## Key differences\n- MSFT grew revenue faster than GOOGL (https://sec.gov/a, https://sec.gov/b)\n"


def test_validate_report():
    assert validate_report(REPORT) is None
    assert validate_report(REPORT + "w " * 800) is None            # under the cap
    too_long = validate_report(REPORT + "w " * 1000)
    assert "limit is 900" in too_long and "Nothing was saved" in too_long
    assert "empty" in validate_report("   ")
    assert "source URLs" in validate_report("# T\n## Summary\nok")


def test_orchestrator_plans_delegates_reads_notes_and_saves_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=lambda q: {
        "query": q, "results": [{"title": "10-K", "url": "https://sec.gov/a", "content": "Revenue up"}]}))
    researcher = ScriptedModel(script=[
        call("tavily_search", query="q", task_id="t1"),
        call("write_file", file_path=PATH, content=GOOD_NOTE),
        AIMessage(content="- Revenue up"),
    ])
    from fin_agent.agents import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "REPORTS_DIR", tmp_path)
    out_file = tmp_path / "msft-googl_final_report.md"  # named after the planned companies
    orch = ScriptedModel(script=[
        call("task", description="too early", subagent_type="researcher"),  # refused: no plan yet
        call("submit_research_plan", companies=["MSFT", "GOOGL"], dimensions=["revenue"],
             tasks=[{"question": "Revenue growth FY2025?", "dimensions": ["revenue"], "why_needed": "asked"}]),
        call("write_todos", todos=[{"content": "research", "status": "in_progress"}]),
        call("task", description="Revenue for MSFT and GOOGL", subagent_type="researcher"),
        call("write_file", file_path="/x.md", content="nope"),          # blocked
        call("save_final_report", markdown=REPORT),                     # refused: note not read yet
        call("read_file", file_path=PATH),
        call("save_final_report", markdown="# too short, no coverage"),  # rejected
        call("save_final_report", markdown=REPORT),
        call("verify_final_report"),                                    # finds: no comparison section
        call("save_final_report", markdown=FIXED),                      # agent corrects the report
        call("verify_final_report"),                                    # clean
        AIMessage(content="done"),
    ])
    request_file = tmp_path / "original_request.md"
    request_file.write_text(
        "# Original Request\n\n## User question\n\ncompare MSFT and GOOGL on revenue\n", encoding="utf-8")
    research_dir = tmp_path / "research"
    (research_dir / "notes").mkdir(parents=True)
    agent = build_orchestrator(
        model=orch, researcher_model=researcher, request_path=request_file,
        backend=FilesystemBackend(root_dir=research_dir, virtual_mode=True),
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "compare MSFT and GOOGL"}]})

    assert orch.idx == len(orch.script) and researcher.idx == len(researcher.script)
    assert (research_dir / "notes" / "msft-googl-revenue.md").is_file()   # note in sandbox
    assert out_file.is_file() and out_file.stat().st_size > 0              # report created, non-empty
    assert out_file.read_text(encoding="utf-8").strip() == FIXED.strip()   # corrected report, outside the sandbox
    assert not (research_dir / "x.md").exists()                            # orchestrator can't write
    by_name = {m.name: m.content for m in out["messages"] if getattr(m, "name", None)}
    assert "write_todos" in by_name and by_name["task"].startswith(f"Notes file: {PATH}")
    assert "Saved final report" in by_name["save_final_report"]
    saves = [m for m in out["messages"] if getattr(m, "name", None) == "save_final_report"]
    assert "read these research notes" in saves[0].content  # synthesis must read the artifact first
    verdicts = [m.content for m in out["messages"] if getattr(m, "name", None) == "verify_final_report"]
    assert "comparison section" in verdicts[0]               # verification caught the gap...
    assert "No obvious gaps" in verdicts[1]                  # ...and the corrected report passes


# ---- dynamic research planning ----------------------------------------------
def task(q, *dims):
    return ResearchTask(question=q, dimensions=list(dims), why_needed="asked")


def plan(planner, dims, tasks, skipped=None):
    return planner.validate(["A", "B"], dims, tasks, skipped)


def test_narrow_request_needs_only_one_task_and_cannot_add_risks():
    p = ResearchPlanner()
    ok = [task("Revenue growth and margins FY2025?", "revenue growth", "profitability")]
    assert plan(p, ["revenue growth", "profitability"], ok) is None            # 1 merged task
    padded = ok + [task("Key risks?", "risks")]
    assert "did not request" in plan(p, ["revenue growth", "profitability"], padded)


def test_broad_request_justifies_several_tasks_but_never_more_than_three():
    p = ResearchPlanner()
    dims = ["revenue", "profitability", "regulatory risks", "outlook"]
    three = [task("q1", "revenue", "profitability"), task("q2", "regulatory risks"), task("q3", "outlook")]
    assert plan(p, dims, three) is None
    four = three + [task("q4", "outlook")]
    assert "hard ceiling 3" in plan(p, dims, four)


def test_requested_dimension_must_be_covered_or_skipped_with_reason():
    p = ResearchPlanner()
    one = [task("q", "revenue")]
    assert "neither covered" in plan(p, ["revenue", "outlook"], one)
    assert plan(p, ["revenue", "outlook"], one, {"outlook": "no guidance published"}) is None


def test_delegation_is_bound_to_the_plan_and_ceiling():
    p = ResearchPlanner()
    assert "submit_research_plan" in p.allow_delegation()           # no plan yet
    p.submit_research_plan.invoke({"companies": ["A"], "dimensions": ["revenue"],
                                   "tasks": [{"question": "q", "dimensions": ["revenue"], "why_needed": "w"}]})
    assert p.allow_delegation() is None                              # the one planned task
    assert "already delegated" in p.allow_delegation()               # a second one is refused
    assert p.delegated_total == 1


def test_decomposition_size_follows_the_request(tmp_path):
    """Same orchestrator, two requests -> a different number of researcher tasks."""
    counts = {}
    for name, dims, n in [("narrow", ["revenue growth", "profitability"], 1),
                          ("broad", ["revenue", "profitability", "regulatory risks", "outlook"], 3)]:
        p = ResearchPlanner()
        groups = [dims[:2], dims[2:3], dims[3:]] if n == 3 else [dims]
        p.submit_research_plan.invoke({"companies": ["A", "B"], "dimensions": dims, "tasks": [
            {"question": f"q{i}", "dimensions": g, "why_needed": "w"} for i, g in enumerate(groups)]})
        counts[name] = sum(p.allow_delegation() is None for _ in range(5))
    assert counts == {"narrow": 1, "broad": 3}


def test_report_is_named_after_the_companies():
    from fin_agent.agents.orchestrator import report_filename
    assert report_filename(["Microsoft", "Alphabet Inc."]) == "microsoft-alphabet-inc_final_report.md"
    assert report_filename([]) == "comparison_final_report.md"


# ---- completeness verification ------------------------------------------------
GOOD = """# Microsoft vs Alphabet
## Executive Summary
- Microsoft grew revenue faster than Alphabet.
## Financial Performance
Microsoft revenue growth was 15% and net margin 36% (https://sec.gov/m). Alphabet revenue growth was 14%
with net margin 28% (https://sec.gov/g).

## Risk Analysis
Microsoft faces regulatory risk (https://ec.europa.eu/t). Alphabet faces antitrust risk (https://justice.gov/g).

## Business Outlook
Microsoft guidance points to cloud growth. Alphabet guidance raises capex; the latest update was not verified.
## Overall Comparison
Microsoft is more profitable; evidence gaps remain.
## Sources
https://sec.gov/m https://sec.gov/g
"""
REQ = "Compare Microsoft and Alphabet on revenue growth, profitability, risks and outlook"


def test_check_report_passes_a_complete_report():
    from fin_agent.report_check import check_report
    assert check_report(GOOD, REQ, ["Microsoft", "Alphabet"], ["revenue growth", "profitability", "risks", "outlook"]) == []


def test_check_report_flags_obvious_gaps():
    from fin_agent.report_check import check_report
    bad = GOOD.replace("Alphabet", "Google").replace("## Sources", "## Notes").replace("https://", "")
    gaps = " | ".join(check_report(bad, REQ, ["Microsoft", "Alphabet"], ["revenue growth", "risks", "outlook"]))
    assert "'Alphabet' never appears" in gaps       # requested company missing
    assert "No source URLs" in gaps                  # sources missing
    assert "Sources section" in gaps                 # obvious section missing
    no_outlook = GOOD.split("## Business Outlook")[0] + "## Sources\nhttps://sec.gov/m https://sec.gov/g\n"
    assert "'outlook'" in " ".join(check_report(no_outlook, REQ, ["Microsoft", "Alphabet"], []))  # factor from the request text alone
