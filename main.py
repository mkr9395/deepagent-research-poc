"""Entry point for the Financial Company Comparison Agent (POC).

The agent is not built yet. For now this file only supports `--check`,
which verifies the environment is ready. Secret values are never printed.
"""
import argparse
import os
import sys
from importlib import metadata

from dotenv import load_dotenv

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Financial Company Comparison Agent")
    parser.add_argument("query", nargs="?", help="e.g. 'Compare Microsoft and Alphabet ...'")
    parser.add_argument("--check", action="store_true", help="verify environment setup")
    args = parser.parse_args()

    if args.check:
        return check()
    print("Agent not implemented yet. Run `python main.py --check` to verify setup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
