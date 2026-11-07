"""Quota tests use a fake search function: no network, no API key, no real calls."""
import logging

import pytest

from fin_agent.tools import tavily_search as ts
from fin_agent.tools.tavily_search import Limits, TavilyGuard

FAKE_KEY = "tvly-SECRET-123"


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        return {
            "query": query,
            "results": [
                {"title": "Blog", "url": "https://randomblog.com/a", "content": "x " * 1000},
                {"title": "10-K", "url": "https://www.sec.gov/10k", "content": "Revenue $1B"},
                {"title": "News", "url": "https://www.reuters.com/n", "content": "News"},
            ],
        }


def make(limits=Limits(), fake=None):
    fake = fake or FakeSearch()
    return TavilyGuard(limits, search_fn=fake), fake


GAP = "Revenue figure conflicts between sources"


def test_ceilings_cap_env(monkeypatch):
    monkeypatch.setattr(ts, "load_dotenv", lambda: None)
    monkeypatch.setenv("TAVILY_MAX_CALLS_PER_TASK", "50")
    monkeypatch.setenv("TAVILY_MAX_RESEARCH_TASKS", "2")
    monkeypatch.setenv("TAVILY_MAX_CALLS_PER_REQUEST", "junk")
    lim = ts.load_limits()
    assert (lim.calls_per_task, lim.research_tasks, lim.calls_per_request) == (5, 2, 15)


def test_request_budget_blocks_and_stops_calling_tavily():
    g, fake = make(Limits(calls_per_task=3, research_tasks=3, calls_per_request=2))
    assert "[TAVILY_BLOCKED" not in g.search("q1", "t1")
    assert "[TAVILY_BLOCKED" not in g.search("q2", "t2")
    out = g.search("q3", "t3")
    assert "REQUEST_BUDGET_EXHAUSTED" in out and "evidence already collected" in out
    assert len(fake.calls) == 2


def test_per_task_limit():
    g, fake = make(Limits(calls_per_task=2, research_tasks=3, calls_per_request=9))
    g.search("q1", "t1")
    g.search("q2", "t1", GAP)
    assert "TASK_BUDGET_EXHAUSTED" in g.search("q3", "t1", GAP)
    assert "[TAVILY_BLOCKED" not in g.search("q4", "t2")  # other task unaffected
    assert len(fake.calls) == 3


def test_task_count_limit():
    g, fake = make(Limits(calls_per_task=3, research_tasks=2, calls_per_request=9))
    g.search("q1", "t1")
    g.search("q2", "t2")
    assert "TASK_LIMIT_REACHED" in g.search("q3", "t3")
    assert "[TAVILY_BLOCKED" not in g.search("q4", "t1", GAP)  # existing task still fine
    assert len(fake.calls) == 3


def test_second_search_needs_gap_reason():
    g, fake = make()
    g.search("q1", "t1")
    assert "GAP_REASON_REQUIRED" in g.search("q2", "t1")
    assert "GAP_REASON_REQUIRED" in g.search("q2", "t1", "more")
    assert len(fake.calls) == 1  # blocked attempts cost nothing
    assert "[TAVILY_BLOCKED" not in g.search("q2", "t1", GAP)


def test_cache_does_not_use_budget():
    g, fake = make()
    g.search("Microsoft revenue FY2025", "t1")
    out = g.search("  microsoft   REVENUE fy2025 ", "t2")
    assert "cached" in out
    assert g.total_calls == 1 and len(fake.calls) == 1


def test_start_request_resets_budget():
    g, _ = make(Limits(1, 1, 1))
    g.search("q1", "t1")
    assert "EXHAUSTED" in g.search("q2", "t1", GAP)
    g.start_request()
    assert "[TAVILY_BLOCKED" not in g.search("q2", "t1")


def test_results_sorted_by_source_tier_and_trimmed():
    g, _ = make()
    out = g.search("q", "t1")
    assert out.index("sec.gov") < out.index("reuters.com") < out.index("randomblog.com")
    assert "[primary]" in out and "[reputable]" in out
    assert len(out) < 2500  # snippets are truncated, no page dumps


def test_logging_and_no_key_leak(caplog, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", FAKE_KEY)

    def boom(query):
        raise RuntimeError(f"401 for key {FAKE_KEY}")

    with caplog.at_level(logging.INFO, logger="fin_agent.tavily"):
        g = TavilyGuard(Limits(), search_fn=boom)
        out = g.search("q", "t1")
    assert "Request Tavily budget: 15" in caplog.text
    assert "Research task 1" in caplog.text
    assert "Tavily call 1/15" in caplog.text
    assert FAKE_KEY not in caplog.text and FAKE_KEY not in out
    assert "TAVILY_ERROR" in out and g.total_calls == 1  # failed attempt still counted


def test_langchain_tool_is_wired(monkeypatch):
    fake = FakeSearch()
    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=fake))
    out = ts.tavily_search.invoke({"query": "q", "task_id": "t1"})
    assert "sec.gov" in out and fake.calls == ["q"]


def test_default_task_ceiling_is_five():
    assert ts.CEILING_CALLS_PER_TASK == 5 and Limits().calls_per_task == 5
    g, fake = make(Limits(calls_per_task=5, research_tasks=3, calls_per_request=15))
    g.search("q1", "t1")
    for i in range(2, 6):
        assert "[TAVILY_BLOCKED" not in g.search(f"q{i}", "t1", GAP)
    assert "TASK_BUDGET_EXHAUSTED" in g.search("q6", "t1", GAP)
    assert len(fake.calls) == 5


def test_source_tier_recognises_company_ir_pages():
    t = ts.source_tier
    assert t("https://www.microsoft.com/en-us/investor/earnings/fy-2026-q4/press-release-webcast") == "primary"
    assert t("https://abc.xyz/investor/events/event-details/2026/x/default.aspx") == "primary"
    assert t("https://s206.q4cdn.com/479360582/files/doc_financials/2025/q4/x.pdf") == "primary"
    assert t("https://investor.apple.com/x") == "primary"
    assert t("https://www.sec.gov/Archives/edgar/x") == "primary"
    assert t("https://www.reuters.com/markets/x") == "reputable"
    assert t("https://www.fool.com/investor/2026/x") == "other"   # /investor/ but not company IR
    assert t("https://ttvalueinvesting.com/research") == "other"
    assert t("https://example.com/blog/investor-tips") == "other"


def test_date_line_only_when_source_provides_one():
    def search(q):
        return {"query": q, "results": [
            {"title": "A", "url": "https://sec.gov/a", "content": "x", "published_date": "2026-04-22"},
            {"title": "B", "url": "https://randomblog.com/b", "content": "y"},
        ]}
    out = TavilyGuard(Limits(), search_fn=search).search("q", "t1")
    assert "Date: 2026-04-22" in out
    assert "date not given" not in out and out.count("Date:") == 1
