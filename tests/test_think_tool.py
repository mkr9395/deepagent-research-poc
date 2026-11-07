from fin_agent.tools.think_tool import think_tool


def test_tool_can_be_invoked_and_stops_when_sufficient():
    out = think_tool.invoke({
        "evidence_summary": "MSFT FY25 revenue $281.7B (sec.gov 10-K).",
        "evidence_sufficient": True,
        "include_in_notes": "Revenue $281.7B - https://www.sec.gov/10k",
    })
    assert "Reflection recorded" in out
    assert "STOP_SEARCHING" in out
    assert "https://www.sec.gov/10k" in out


def test_justified_search_suggests_gap_reason():
    out = think_tool.invoke({
        "evidence_summary": "Only revenue found.",
        "evidence_sufficient": False,
        "important_gap": "No operating margin for Alphabet",
        "second_search_justified": True,
    })
    assert "SEARCH_AGAIN" in out and "No operating margin for Alphabet" in out


def test_search_without_gap_is_rejected():
    out = think_tool.invoke({
        "evidence_summary": "Some data.",
        "evidence_sufficient": False,
        "second_search_justified": True,
    })
    assert "STOP_SEARCHING" in out and "NOT justified" in out


def test_conflict_is_flagged():
    out = think_tool.invoke({
        "evidence_summary": "Two figures.",
        "evidence_sufficient": False,
        "sources_conflict": "10-K says $245B, blog says $262B",
        "second_search_justified": True,
    })
    assert "SEARCH_AGAIN" in out and "Conflict" in out
