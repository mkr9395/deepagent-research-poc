"""Entry point for the Financial Company Comparison Agent (POC).

    python main.py "Compare Microsoft and Alphabet on revenue growth, ..."
    python main.py --check        # verify the environment (no secret values printed)

Add -v to log the planning / delegation / context hand-offs as they happen.
"""
import argparse
import logging
import os
import sys
from importlib import metadata

from dotenv import load_dotenv

from fin_agent.agents.orchestrator import build_orchestrator
from fin_agent.agents.researcher import get_backend, RESEARCH_DIR
from fin_agent.request import save_original_request
from fin_agent.tools.tavily_search import start_request

PACKAGES = ["deepagents", "langchain-openai", "tavily-python", "python-dotenv", "pytest"]
REQUIRED_VARS = ["OPENAI_API_KEY", "TAVILY_API_KEY"]


def check() -> int:
    load_dotenv()
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for pkg in PACKAGES:
        try:
            print(f"  [ok]      {pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"  [missing] {pkg}")
            ok = False
    for var in REQUIRED_VARS:
        is_set = bool(os.environ.get(var))
        print(f"  [{'set' if is_set else 'missing'}]".ljust(12) + var)
        ok = ok and is_set
    print("Environment ready." if ok else "Environment NOT ready.")
    return 0 if ok else 1


def reset_run_artifacts() -> None:
    """Start each run with empty notes so the orchestrator can never read stale ones.
    Final reports are named per company pair and are never deleted."""
    for note in (RESEARCH_DIR / "notes").glob("*.md"):
        note.unlink()


def run(query: str) -> int:
    load_dotenv()
    saved = save_original_request(query)
    print(f"Saved original request to {saved}")
    get_backend()  # ensure outputs/research/notes exists
    reset_run_artifacts()
    start_request()  # reset the Tavily budget for this request

    agent = build_orchestrator()
    print("Running agent (planning, researching, synthesizing - this can take a few minutes)...")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 100},
    )
    print("\n" + result["messages"][-1].text)
    saved = [
        m.text.split(" to ", 1)[1].rsplit(" (", 1)[0]
        for m in result["messages"]
        if getattr(m, "name", None) == "save_final_report" and m.text.startswith("Saved final report to ")
    ]
    if saved:
        print(f"\nFinal report: {saved[-1]}")
        return 0
    print("\nNo final report was saved.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Financial Company Comparison Agent")
    parser.add_argument("query", nargs="?", help="e.g. 'Compare Microsoft and Alphabet ...'")
    parser.add_argument("--check", action="store_true", help="verify environment setup")
    parser.add_argument("-v", "--verbose", action="store_true", help="log searches and context hand-offs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if args.check:
        return check()
    if not args.query:
        parser.print_help()
        return 2
    return run(args.query)


if __name__ == "__main__":
    sys.exit(main())
