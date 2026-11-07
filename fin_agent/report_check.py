"""Lightweight completeness check of the final report against the original request.

Not an evaluation framework and not an agent: a few keyword/structure checks that catch
OBVIOUS gaps (a company or requested factor that never appears, no sources, no comparison
section). The main agent calls it through the `verify_final_report` tool, reads the gap list
and corrects the report where practical. Findings are advisory: a keyword miss can be a false
alarm, in which case the agent leaves the report as is.
"""
from __future__ import annotations

import re

# label -> (regex that, when found in the REQUEST, means the user asked for it,
#           regex that the REPORT must match in a paragraph that also names each company)
FACTORS: dict[str, tuple[str, str]] = {
    "revenue growth": (r"revenue|\bsales\b|growth", r"revenue|\bsales\b|growth"),
    "profitability": (r"profit|margin|income|earnings", r"profit|margin|net income|operating income|earnings"),
    "risks": (r"\brisks?\b", r"\brisks?\b"),
    "outlook": (r"outlook|guidance|forecast|future|prospects", r"outlook|guidance|expects|forecast|future"),
}

# Structure expected in any comparison report: (what is missing, regex for a heading line).
SECTIONS = [
    ("an executive summary / summary section", r"summary"),
    ("a comparison section (key differences / overall comparison)", r"comparison|differences"),
    ("a Sources section", r"sources|references"),
]
UNCERTAINTY = r"gap|uncertain|not found|not verified|unverified|not available|limitation|could not"


def request_question(request_markdown: str) -> str:
    """The user's question out of the saved request artifact (falls back to the whole text)."""
    m = re.search(r"##\s*User question\s*\n(.*?)(?=\n##\s|\Z)", request_markdown, re.S | re.I)
    return (m.group(1) if m else request_markdown).strip()


def _company_token(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return words[0] if words else ""


def _requested_factors(request: str, dimensions: list[str]) -> dict[str, str]:
    """label -> report regex, from the request text and the plan's dimensions."""
    wanted: dict[str, str] = {}
    texts = [request] + list(dimensions or [])
    for label, (trigger, needs) in FACTORS.items():
        if any(re.search(trigger, t, re.I) for t in texts):
            wanted[label] = needs
    for dim in dimensions or []:  # a dimension outside the table: look for its own words
        if not any(re.search(t, dim, re.I) for t, _ in FACTORS.values()):
            words = [w for w in re.findall(r"[a-z]{4,}", dim.lower())]
            if words:
                wanted[dim] = "|".join(re.escape(w) for w in words)
    return wanted


def check_report(report: str, request: str, companies: list[str], dimensions: list[str]) -> list[str]:
    """Return a list of obvious gaps (empty list = nothing obvious missing)."""
    gaps: list[str] = []
    low = report.lower()
    paragraphs = [p.lower() for p in re.split(r"\n\s*\n", report) if p.strip()]
    headings = [re.sub(r"[^a-z ]", " ", l.lstrip("#").lower()) for l in report.splitlines() if l.lstrip().startswith("#")]
    tokens = {c: _company_token(c) for c in companies or [] if _company_token(c)}

    for company, tok in tokens.items():
        if tok not in low:
            gaps.append(f"Company not covered: '{company}' never appears in the report.")

    for label, needs in _requested_factors(request, dimensions).items():
        if not re.search(needs, low):
            gaps.append(f"Requested factor not addressed: '{label}'.")
            continue
        for company, tok in tokens.items():
            if tok in low and not any(tok in p and re.search(needs, p) for p in paragraphs):
                gaps.append(f"'{label}' is not addressed for {company} (no passage links them).")

    urls = set(re.findall(r"https?://[^\s)\]>]+", report))
    if not urls:
        gaps.append("No source URLs in the report.")
    elif len(urls) < max(2, len(tokens)):
        gaps.append(f"Only {len(urls)} distinct source URL(s) for {max(2, len(tokens))}+ companies/claims; check each figure is sourced.")

    for missing, pattern in SECTIONS:
        if not any(re.search(pattern, h) for h in headings):
            gaps.append(f"Missing section: {missing}.")

    if not re.search(UNCERTAINTY, low):
        gaps.append("No statement of uncertainty or evidence gaps anywhere in the report.")
    return gaps
