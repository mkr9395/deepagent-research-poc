"""Standalone live check of the Tavily integration (no agent involved).

Run:  python -m fin_agent.tools.check_tavily

Makes exactly TAVILY_MAX_CALLS_PER_REQUEST real searches, then one more attempt
that must be blocked by the budget (no Tavily call). The API key is only
reported as set/missing; its value is never printed.
"""
import logging
import os
import sys

from dotenv import load_dotenv

from fin_agent.tools.tavily_search import get_guard, tavily_search

SAFETY_CAP = 5  # refuse to run a live check bigger than this
QUERIES = [
    "Microsoft fiscal year 2025 annual revenue investor relations",
    "Alphabet 2025 annual revenue 10-K sec.gov",
    "Microsoft Azure growth rate latest quarter",
    "Alphabet Google Cloud operating income latest quarter",
    "Microsoft Alphabet capital expenditure 2025",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("1. API key")
    check("TAVILY_API_KEY loaded (value hidden)", bool(os.environ.get("TAVILY_API_KEY")))
    if failures:
        return 1

    guard = get_guard()
    lim = guard.limits
    print(f"\nEffective limits: {lim.calls_per_request} calls/request, "
          f"{lim.calls_per_task} calls/task, {lim.research_tasks} tasks")
    if not 1 <= lim.calls_per_request <= SAFETY_CAP:
        print(f"Refusing live run: set TAVILY_MAX_CALLS_PER_REQUEST between 1 and {SAFETY_CAP}, e.g. 3.")
        return 1

    guard.start_request()
    print(f"\n2. Live searches ({lim.calls_per_request} real calls)")
    for i in range(lim.calls_per_request):
        task = f"check-{i // lim.calls_per_task + 1}"
        gap = "" if i % lim.calls_per_task == 0 else "Live check: confirming a second search is allowed with a reason"
        out = tavily_search.invoke({"query": QUERIES[i], "task_id": task, "gap_reason": gap})
        ok = out.startswith("Search results for") and "URL: http" in out
        check(f"search {i + 1}/{lim.calls_per_request} returned results", ok, "" if ok else out[:200])

    print("\n3. Usage counter")
    check("counter equals configured limit", guard.total_calls == lim.calls_per_request,
          f"{guard.total_calls}/{lim.calls_per_request}")

    print("\n4. Limit respected")
    out = tavily_search.invoke({"query": "one more search that must be refused", "task_id": "check-extra",
                                "gap_reason": "Live check: this call must be blocked"})
    check("extra call blocked", "[TAVILY_BLOCKED:REQUEST_BUDGET_EXHAUSTED]" in out)
    check("blocked call did not increase counter", guard.total_calls == lim.calls_per_request,
          f"{guard.total_calls}/{lim.calls_per_request}")

    print("\nALL CHECKS PASSED" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
