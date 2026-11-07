"""Tavily web-search tool with application-level budget enforcement.

The limits live in code (`TavilyGuard`), not in prompts: an over-eager agent
simply gets a "blocked" message back instead of a Tavily call.

Limits (env-configurable, clamped to the hard ceilings below):
  TAVILY_MAX_CALLS_PER_TASK     searches per research task      (ceiling 5)
  TAVILY_MAX_RESEARCH_TASKS     research tasks per user request (ceiling 3)
  TAVILY_MAX_CALLS_PER_REQUEST  searches per user request       (ceiling 15)

A "research task" is identified by the `task_id` the agent passes to the tool
(e.g. "msft-revenue"). The first search of a task is free of conditions; the
2nd onward must carry a `gap_reason` explaining the gap/conflict.

The API key is read from the environment only and is never logged or returned.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.tools import tool

logger = logging.getLogger("fin_agent.tavily")

# Hard ceilings: env vars can lower these, never raise them.
CEILING_CALLS_PER_TASK = 5
CEILING_RESEARCH_TASKS = 3
CEILING_CALLS_PER_REQUEST = 15

MAX_RESULTS = 4  # CLAUDE.md: 3-5 results
SNIPPET_CHARS = 600  # no raw page dumps
MIN_GAP_REASON_CHARS = 15

# Reliability tiers for financial research (used to sort + label results).
REPUTABLE_PRESS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com",
    "barrons.com", "marketwatch.com", "morningstar.com", "economist.com",
    "apnews.com", "fortune.com",
}
# Investor-relations CDN/hosting domains used by many companies for filings and releases.
IR_HOSTING = {"q4cdn.com", "q4web.com", "investorroom.com"}
# Sites with an /investor/ section that is NOT the company's own IR page.
NOT_PRIMARY = {"fool.com", "investopedia.com", "seekingalpha.com", "yahoo.com", "tickeron.com"}
_IR_SEGMENTS = {"investor", "investors", "investor-relations", "investorrelations", "ir"}
EXCLUDED_DOMAINS = [
    "reddit.com", "quora.com", "pinterest.com", "facebook.com",
    "x.com", "twitter.com", "tiktok.com", "youtube.com",
]


@dataclass(frozen=True)
class Limits:
    calls_per_task: int = CEILING_CALLS_PER_TASK
    research_tasks: int = CEILING_RESEARCH_TASKS
    calls_per_request: int = CEILING_CALLS_PER_REQUEST


def _env_int(name: str, ceiling: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ceiling
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, ceiling)
        return ceiling
    return max(0, min(value, ceiling))


def load_limits() -> Limits:
    """Read limits from env (after loading .env), clamped to the hard ceilings."""
    load_dotenv()
    return Limits(
        calls_per_task=_env_int("TAVILY_MAX_CALLS_PER_TASK", CEILING_CALLS_PER_TASK),
        research_tasks=_env_int("TAVILY_MAX_RESEARCH_TASKS", CEILING_RESEARCH_TASKS),
        calls_per_request=_env_int("TAVILY_MAX_CALLS_PER_REQUEST", CEILING_CALLS_PER_REQUEST),
    )


def _host_in(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def source_tier(url: str) -> str:
    """Classify a URL: primary (company IR / SEC / regulators), reputable press, or other.

    Heuristic: primary = government/regulator hosts, `investor.`/`ir.` hosts, IR hosting
    CDNs (e.g. q4cdn.com), or an `/investor(s)/` section within the first two path
    segments (e.g. microsoft.com/en-us/investor/..., abc.xyz/investor/...).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if _host_in(host, NOT_PRIMARY):
        return "other"
    segments = [x for x in parsed.path.lower().split("/") if x][:2]
    if (
        host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".europa.eu")
        or host.startswith(("investor.", "investors.", "ir."))
        or "investor" in host.split(".")[0]
        or _host_in(host, IR_HOSTING)
        or any(seg in _IR_SEGMENTS for seg in segments)
    ):
        return "primary"
    if _host_in(host, REPUTABLE_PRESS):
        return "reputable"
    return "other"


_TIER_ORDER = {"primary": 0, "reputable": 1, "other": 2}


def _blocked(code: str, detail: str) -> str:
    return (
        f"[TAVILY_BLOCKED:{code}] {detail} "
        "Do not retry. Continue with the evidence already collected and flag "
        "any gaps as unverified in your notes."
    )


class TavilyGuard:
    """Enforces the per-task / per-request budget around a search function."""

    def __init__(self, limits: Limits | None = None, search_fn: Callable[..., dict] | None = None):
        self.limits = limits or load_limits()
        self._search_fn = search_fn or self._real_search
        self._lock = threading.Lock()
        self.start_request()

    # ---- request lifecycle -------------------------------------------------
    def start_request(self) -> None:
        """Reset all counters. Call once at the start of each user request."""
        with self._lock:
            self.total_calls = 0
            self.task_calls: dict[str, int] = {}  # insertion order = task number
            self._cache: dict[str, str] = {}
        logger.info(
            "Request Tavily budget: %d (max %d research tasks, %d calls/task)",
            self.limits.calls_per_request, self.limits.research_tasks, self.limits.calls_per_task,
        )

    # ---- the enforcement point --------------------------------------------
    def search(self, query: str, task_id: str, gap_reason: str = "") -> str:
        query, task_id, gap_reason = query.strip(), task_id.strip(), gap_reason.strip()
        if not query or not task_id:
            return _blocked("BAD_INPUT", "Both `query` and `task_id` are required.")
        cache_key = " ".join(query.lower().split())

        with self._lock:
            if cache_key in self._cache:
                logger.info("Tavily cache hit (no call used): %r", query)
                return self._cache[cache_key] + "\n(cached result; no budget used)"

            lim = self.limits
            if self.total_calls >= lim.calls_per_request:
                logger.warning("Tavily request budget exhausted (%d/%d)", self.total_calls, lim.calls_per_request)
                return _blocked(
                    "REQUEST_BUDGET_EXHAUSTED",
                    f"The request-level search budget ({lim.calls_per_request} calls) is used up.",
                )

            is_new_task = task_id not in self.task_calls
            if is_new_task and len(self.task_calls) >= lim.research_tasks:
                logger.warning("Research task limit reached; refusing new task %r", task_id)
                return _blocked(
                    "TASK_LIMIT_REACHED",
                    f"Already used {lim.research_tasks} research tasks; '{task_id}' would be another.",
                )

            used_in_task = self.task_calls.get(task_id, 0)
            if used_in_task >= lim.calls_per_task:
                logger.warning("Task %r reached its %d-call limit", task_id, lim.calls_per_task)
                return _blocked(
                    "TASK_BUDGET_EXHAUSTED",
                    f"Task '{task_id}' already used its {lim.calls_per_task} searches.",
                )

            if used_in_task >= 1 and len(gap_reason) < MIN_GAP_REASON_CHARS:
                return _blocked(
                    "GAP_REASON_REQUIRED",
                    "A further search in the same task needs `gap_reason` naming the "
                    "important missing information or conflicting figures.",
                )

            # Reserve the slot (counted even if Tavily later errors: it may be billed).
            self.task_calls[task_id] = used_in_task + 1
            self.total_calls += 1
            task_no = list(self.task_calls).index(task_id) + 1
            call_no = self.total_calls
            if is_new_task:
                logger.info("Research task %d (%s)", task_no, task_id)
            logger.info(
                "Tavily call %d/%d (task %d: %d/%d)%s | query=%r",
                call_no, lim.calls_per_request, task_no, used_in_task + 1, lim.calls_per_task,
                f" | gap: {gap_reason}" if gap_reason else "", query,
            )

        # Network call happens outside the lock.
        try:
            raw = self._search_fn(query)
            text = self._format(raw, call_no, task_id)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the agent
            msg = self._redact(f"{type(exc).__name__}: {exc}")
            logger.error("Tavily call %d failed: %s", call_no, msg)
            return f"[TAVILY_ERROR] Search failed ({msg}). This attempt still counted toward the budget."

        with self._lock:
            self._cache[cache_key] = text
        return text

    # ---- helpers -----------------------------------------------------------
    def _format(self, raw: dict, call_no: int, task_id: str) -> str:
        results = sorted(
            raw.get("results", []),
            key=lambda r: _TIER_ORDER[source_tier(r.get("url", ""))],
        )
        remaining = self.limits.calls_per_request - self.total_calls
        lines = [
            f"Search results for: {raw.get('query', '')}",
            f"(task '{task_id}' | request calls used {call_no}/{self.limits.calls_per_request}, {remaining} left)",
        ]
        if not results:
            lines.append("No results.")
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            snippet = " ".join((r.get("content") or "").split())[:SNIPPET_CHARS]
            lines += [
                f"{i}. {r.get('title', '(untitled)')} [{source_tier(url)}]",
                f"   URL: {url}",
            ]
            if r.get("published_date"):  # only shown when the source provides one
                lines.append(f"   Date: {r['published_date']}")
            lines.append(f"   {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _redact(text: str) -> str:
        key = os.environ.get("TAVILY_API_KEY", "")
        return text.replace(key, "[REDACTED]") if key else text

    def _real_search(self, query: str) -> dict:
        from tavily import TavilyClient

        load_dotenv()
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        return TavilyClient(api_key=key).search(
            query=query,
            search_depth="basic",
            max_results=MAX_RESULTS,
            exclude_domains=EXCLUDED_DOMAINS,
        )


# ---- module-level singleton + LangChain tool -------------------------------
_guard: TavilyGuard | None = None


def get_guard() -> TavilyGuard:
    global _guard
    if _guard is None:
        _guard = TavilyGuard()
    return _guard


def start_request() -> None:
    """Reset the Tavily budget. The entry point calls this once per user request."""
    get_guard().start_request()


@tool
def tavily_search(query: str, task_id: str, gap_reason: str = "") -> str:
    """Search the web (Tavily, basic depth) for financial research.

    Args:
        query: Specific search query. Prefer company investor-relations pages,
            SEC filings (10-K/10-Q/8-K), regulators, and reputable financial
            press (Reuters, Bloomberg, FT, WSJ). Include company, metric and fiscal period.
        task_id: Short stable id of YOUR research task, e.g. "msft-revenue".
            Reuse the same id for every search within that task.
        gap_reason: Required from the 2nd search of a task onward: the important
            information gap or conflicting figures the extra search resolves.

    Budget is enforced in code: 5 searches/task, 3 tasks and 15 searches per
    request at most. Results carry a source tier (primary / reputable / other),
    URL and date. If the response starts with [TAVILY_BLOCKED], do not retry;
    work from the evidence you already have and flag gaps as unverified.
    """
    return get_guard().search(query, task_id, gap_reason)
