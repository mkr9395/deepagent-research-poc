# Financial Company Comparison Agent (POC)

A one-day proof of concept built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents).
Ask for a comparison of two companies; an orchestrator agent plans the research, delegates focused
tasks to a reusable research subagent (Tavily web search), keeps detailed notes on the filesystem,
and saves a markdown comparison report.

> Status: project scaffold only. The agent is not implemented yet.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then add your own keys to .env
python main.py --check
```

`.env` is git-ignored. Never commit real keys.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point |
| `fin_agent/agents/` | Orchestrator and researcher subagent definitions |
| `fin_agent/tools/` | Tools, e.g. the budgeted Tavily search wrapper |
| `outputs/research/` | Agent workspace: `notes/*.md` and `report.md` (contents git-ignored) |
| `tests/` | Light smoke tests (`python -m pytest -q`) |
| `CLAUDE.md` | Project guidelines |
