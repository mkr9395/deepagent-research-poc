# CLAUDE.md

## Project Objective

A one-day POC: a **Financial Company Comparison Agent** built on LangChain Deep Agents.
Given a request like "Compare Microsoft and Alphabet on revenue growth, profitability, risks and outlook", the agent plans the research, delegates focused research to a subagent, uses Tavily web search, keeps detailed findings on the filesystem, and saves a final markdown comparison report.

Keep it a practical POC. Do not build production architecture.

## Tech Approach

- Python with the standard `venv` and `pip` (the venv is `.venv/`). **Do not use uv.**
- **LangChain Deep Agents (`deepagents`, `create_deep_agent`) is the core framework.** Use its built-in planning (`write_todos`), subagent delegation (`task`) and filesystem tools. Do not replace them with a hand-built LangGraph workflow.
- Models: OpenAI, small and cheap (default `gpt-6.1-terra`), set via env vars (`ORCHESTRATOR_MODEL`, `RESEARCHER_MODEL`).
- Web search: Tavily, through a thin wrapper tool we control.
- Dependencies are tracked in `requirements.txt`.

## Agent Design

- **One orchestrator + one reusable `researcher` subagent definition.** The orchestrator invokes the researcher multiple times with different briefs (company + topic). Do not create three permanently separate agents.
- **Filesystem/context management is a core part of the design.** Detailed research goes to files under `workspace/` (e.g. `workspace/notes/*.md`); the orchestrator keeps only short summaries and file paths in its context, reads notes at synthesis time, and writes `workspace/report.md`. The filesystem backend must be sandboxed to `workspace/`.
- The researcher returns a brief summary plus the notes file path, not the full findings.

## Research and Output Rules

- **Tavily limits:** hard per-run search budget enforced in the wrapper (about 12), 2-3 searches per research task, `search_depth="basic"`, 3-5 results, query cache for duplicates, log each search and the running count.
- **Each research task's output is limited to 350 words.**
- **The final report is not limited to 500 words**; make it as long as a useful comparison needs.
- **No raw webpage dumps.** Store extracted facts and short summaries only.
- **Preserve source references:** every figure or claim in notes and in the report keeps its source URL (and date where available). Flag anything unverified or stale.

## Out of Scope

Avoid vector databases, RAG, FastAPI, any UI, Docker, Kubernetes, and any database infrastructure. Do not add them, and do not propose them unless asked.

## Secrets

Never read, print or inspect `.env`, `.env.*` or any file containing secrets or API keys. Required variable names are `OPENAI_API_KEY` and `TAVILY_API_KEY`. Only run value-free presence checks, and ask first. Load keys via `python-dotenv`/environment and never log them. `.env.example` has names with empty values only.

## Testing

- Test each major feature before moving to the next one, but keep it light.
- Prefer a quick smoke test or a small script over a test framework setup. A few simple assertions are enough.
- Do not over-engineer tests, mocks or CI.

## Development Workflow

We work incrementally:

1. Implement one feature.
2. Explain the change.
3. Run a verification command.
4. Inspect the result.
5. Then continue to the next feature.

Do not implement application functionality until the current step has been asked for.
