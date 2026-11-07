"""Offline smoke test of the LLM usage tracker (scripted models, fake Tavily)."""
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage

from fin_agent.agents import orchestrator as orch_mod
from fin_agent.agents.orchestrator import build_orchestrator
from fin_agent.tools import tavily_search as ts
from fin_agent.tools.tavily_search import Limits, TavilyGuard
from fin_agent.usage import UsageTracker
from tests.test_researcher import GOOD_NOTE, ScriptedModel, call

PATH = "/notes/usage-check.md"


def with_usage(msg: AIMessage, i=100, o=10) -> AIMessage:
    msg.usage_metadata = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
    return msg


def test_usage_tracker_counts_calls_and_tokens_per_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=lambda q: {
        "query": q, "results": [{"title": "t", "url": "https://sec.gov/a", "content": "x"}]}))
    monkeypatch.setattr(orch_mod, "REPORTS_DIR", tmp_path)
    research = tmp_path / "research"
    (research / "notes").mkdir(parents=True)

    researcher = ScriptedModel(script=[
        with_usage(call("tavily_search", query="q", task_id="t1")),
        with_usage(call("write_file", file_path=PATH, content=GOOD_NOTE)),
        with_usage(AIMessage(content="- done")),
    ])
    orch = ScriptedModel(script=[
        with_usage(call("submit_research_plan", companies=["A", "B"], dimensions=["risks"],
                        tasks=[{"question": "Risks?", "dimensions": ["risks"], "why_needed": "asked"}]), 1000, 50),
        with_usage(call("task", description="Risks for A and B", subagent_type="researcher"), 1000, 50),
        with_usage(AIMessage(content="stop"), 1000, 50),
    ])
    agent = build_orchestrator(model=orch, researcher_model=researcher,
                               backend=FilesystemBackend(root_dir=research, virtual_mode=True))
    usage = UsageTracker()
    agent.invoke({"messages": [{"role": "user", "content": "compare A and B on risks"}]},
                 config={"callbacks": [usage]})

    s = usage.stats
    print(usage.summary())
    assert sum(v["started"] for v in s.values()) == 6           # 3 orchestrator + 3 researcher calls
    assert all(v["started"] == v["finished"] for v in s.values())
    assert sum(v["input"] for v in s.values()) == 3 * 1000 + 3 * 100
    assert "orchestrator" in s and len(s) >= 2                  # researcher calls attributed separately
    assert "TOTAL" in usage.summary()
