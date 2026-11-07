"""Lightweight reflection tool for the research subagent.

Not an LLM and not an agent: it takes the model's own structured assessment,
applies a few deterministic consistency checks, logs it, and echoes back a
short decision the model can act on. The value is that the model must fill in
specific fields (sufficiency, gap, conflicts, what to keep) before deciding
whether to spend another Tavily search.
"""
import logging

from langchain_core.tools import tool

logger = logging.getLogger("fin_agent.think")


@tool
def think_tool(
    evidence_summary: str,
    evidence_sufficient: bool,
    important_gap: str = "",
    sources_conflict: str = "",
    second_search_justified: bool = False,
    include_in_notes: str = "",
) -> str:
    """Pause and reflect after a search, before deciding the next step.

    Use after each Tavily result and before any additional search. Be concrete.

    Args:
        evidence_summary: What you have found so far, with sources (1-3 sentences).
        evidence_sufficient: True if the evidence already answers the research brief.
        important_gap: The specific missing fact that matters to the comparison
            (empty if none). Minor or nice-to-have gaps do not count.
        sources_conflict: Conflicting figures/claims and which sources disagree
            (empty if none).
        second_search_justified: True only if an important gap or conflict remains
            AND one more targeted search could plausibly resolve it.
        include_in_notes: Facts, source URLs/dates and caveats (unverified/stale)
            to write into the notes file.
    """
    gap, conflict = important_gap.strip(), sources_conflict.strip()
    notes = []

    if second_search_justified and not (gap or conflict):
        notes.append("Search NOT justified: no important gap or conflict was stated.")
        second_search_justified = False
    if evidence_sufficient and second_search_justified:
        notes.append("Note: evidence marked sufficient yet another search requested; confirm the gap really matters.")
    if gap and evidence_sufficient:
        notes.append("Gap listed although evidence is marked sufficient; record the gap as a caveat in the notes.")
    if conflict:
        notes.append("Conflict: report both figures with their sources and flag as unverified unless resolved.")

    if second_search_justified:
        decision = f"SEARCH_AGAIN. Use as gap_reason: {(gap or conflict)[:200]}"
    else:
        decision = "STOP_SEARCHING. Write the notes file from the evidence you have; flag gaps as unverified."

    logger.info("Reflection | sufficient=%s | gap=%r | conflict=%r | decision=%s",
                evidence_sufficient, gap, conflict, decision.split(".")[0])

    lines = ["Reflection recorded.", f"Decision: {decision}"]
    lines += notes
    if include_in_notes.strip():
        lines.append(f"Include in notes: {include_in_notes.strip()}")
    return "\n".join(lines)
