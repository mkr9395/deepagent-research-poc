# Architecture

A one-day POC: a **Financial Company Comparison Agent** on LangChain Deep Agents.
Given a request such as *"Compare JPMorgan Chase and Bank of America on revenue growth,
profitability, risks and outlook"*, an orchestrator plans the work, delegates focused
research to a reusable `researcher` subagent, keeps detailed findings on the filesystem,
and writes a markdown comparison report.

Status legend: **[built]** exists and is tested, **[planned]** not implemented yet.

## 1. System overview

```
 user request
      │
      ▼
 main.py ── save_original_request() ──▶ outputs/original_request.md        [built]
      │                                  └──▶ outputs/requests/<timestamp>.md  [built]
      │      tavily start_request()  (reset per-request search budget)     [built hook]
      ▼
 ┌──────────────────────────────────────────────┐
 │ Orchestrator  (create_deep_agent)            │                          [planned]
 │  write_todos · task · read_file · write_file │
 └───────────────┬──────────────────────────────┘
                 │ task(subagent_type="researcher", brief)   ×N (≤ 3)
                 ▼
 ┌──────────────────────────────────────────────┐
 │ researcher subagent  (one definition,        │                          [built]
 │ reused per brief; isolated context)          │
 │  tavily_search · think_tool · write_file     │
 └───────┬──────────────────────┬───────────────┘
         │ search (≤5/task)     │ write_file /notes/<slug>.md  (≤500 words)
         ▼                      ▼
   TavilyGuard ──▶ Tavily    outputs/research/notes/<slug>.md
                                   ▲
                                   │ read_file at synthesis time
                 orchestrator ─────┘── write_file /report.md (≤700 words)
                                       → outputs/research/report.md        [planned]
```

The parent's context holds only short summaries and note paths. Detailed, sourced findings
live on disk.

## 2. Components

| Path | Role | Status |
|---|---|---|
| `main.py` | CLI. `--check` verifies packages and that required env vars are *set* (values never printed). A query saves the original request; the agent run is not wired yet. | partial |
| `fin_agent/request.py` | `save_original_request()` writes the latest request to `outputs/original_request.md` (overwritten each run) and a timestamped copy to `outputs/requests/` (history, never overwritten; same-second collisions get a `_2`, `_3` suffix). | built |
| `fin_agent/tools/tavily_search.py` | `TavilyGuard` + `tavily_search` LangChain tool; budget enforced in code. | built |
| `fin_agent/tools/think_tool.py` | Structured reflection tool with deterministic consistency checks. | built |
| `fin_agent/tools/check_tavily.py` | Standalone live check of the Tavily integration (spends real calls). | built |
| `fin_agent/research_guard.py` | `ResearchGuardMiddleware`: note path/size/section checks, result truncation. | built |
| `fin_agent/agents/researcher.py` | Researcher prompt, `get_researcher_subagent()`, `build_researcher_agent()`, sandboxed backend, `python -m` smoke runner. | built |
| `fin_agent/agents/orchestrator.py` | Orchestrator definition and prompt. | planned |
| `tests/` | Light offline tests (fake models, fake search). | built |
| `outputs/` | Runtime workspace (contents git-ignored). | — |

## 3. Tavily search layer

`tavily_search(query, task_id, gap_reason="")` delegates to a singleton `TavilyGuard`.
All limits live in code, so a misbehaving model gets a `[TAVILY_BLOCKED:<code>]` message
instead of a network call.

- **Limits** (env can lower, never raise): 5 searches per task, 3 research tasks and 15
  searches per request. A "task" is the `task_id` the model passes.
- **Gap reason:** the 2nd search onward in a task must carry a `gap_reason` (≥15 chars).
- **Cache:** duplicate queries (normalised) return the cached result and use no budget.
- **Failed calls still count** (they may be billed).
- **Request lifecycle:** `start_request()` resets counters; the entry point must call it once
  per user request.
- **Search shape:** `search_depth="basic"`, 4 results, social/forum domains excluded,
  snippets cut to 600 chars (no page dumps). Results are sorted by source tier
  (primary: IR/SEC/regulators, then reputable press, then other) and carry URL and date.
- **Logging:** every call logs request count and task count; the API key is read from the
  environment only, redacted from errors, never logged.

## 4. The researcher subagent

One `SubAgent` spec (`name="researcher"`) reused for every brief: performance, risk,
outlook, anything else. The orchestrator invokes it via the Deep Agents `task` tool, once
per focused brief; each run has an isolated context containing only the brief.

Workflow (prompt-driven, tool-guarded):

1. Choose a `task_id` slug; use it for every search and as the note filename.
2. `tavily_search` → `think_tool` on the result.
3. Search again only if `think_tool` returns `SEARCH_AGAIN` (important gap or conflict),
   passing its `gap_reason`. Never more than 5.
4. `write_file /notes/<task_id>.md` with five sections: Research Question, Key Findings,
   Financial Facts, Sources, Caveats.
5. Reply with `Notes file: /notes/<task_id>.md` plus 3–6 headline bullets.

Model: `RESEARCHER_MODEL` env var, default `gpt-6.1-terra` (`openai:` prefix added if missing).

## 5. Filesystem and context management

- **Backend:** Deep Agents `FilesystemBackend(root_dir=outputs/research, virtual_mode=True)`.
  The agent sees `/` as `outputs/research/`, so it cannot reach `.env` or other repo files.
  `/notes/x.md` is `outputs/research/notes/x.md` on disk.
- **Writes** use the built-in `write_file` tool; no custom file code.
- **Why:** detailed findings and source URLs stay out of the orchestrator's context; the parent
  receives roughly 250 words per task; notes persist for inspection and for exact-figure
  re-reading at synthesis time.
- **Consumption:** the orchestrator keeps each `Notes file:` path and bullets, reads the notes
  with `read_file` when synthesising, then writes `/report.md` keeping source URLs and caveats.

### Context flow between orchestrator and researcher

```
orchestrator ──task(description=brief)──▶ researcher   (mode="isolated")
                                           sees: its system prompt + the brief ONLY
                                           │ search results, think_tool, drafts stay here
                                           ▼
                                  write_file /notes/<slug>.md   (full detail, on disk)
                                           │
orchestrator ◀── one ToolMessage ──────────┘
   "Notes file: /notes/<slug>.md" + ≤500-word summary
   (state key `messages` only; nothing else is returned)
```

- Parent → child: only the `description` string. The parent's conversation, todos and prior
  results are not passed (`mode="isolated"`, set explicitly on the spec).
- Child → parent: only the child's final message. Raw search text, `think_tool` output and the
  note body never enter the parent's context; they are read on demand with `read_file`.
- The artifact reference is guaranteed by `ResearchGuardMiddleware.after_agent`, not by the prompt.
- Covered by `tests/test_researcher.py` (context-flow tests with marker strings).
- Visible in the logs (logger `fin_agent.context`, INFO): one `PARENT -> RESEARCHER` line (brief size,
  messages received) and `RESEARCHER -> PARENT` lines (reply size, note path and size, and how much
  stayed behind). This is plain logging inside the existing guard, not a context-management framework.

## 6. Code-enforced output limits

`ResearchGuardMiddleware` (attached to the researcher) turns prompt requests into guarantees:

| Rule | Enforcement |
|---|---|
| Note path must match `/notes/<slug>.md` | `write_file` rejected otherwise |
| Note ≤ 500 words (`NOTE_MAX_WORDS`) | `write_file` rejected with the overage; nothing saved; model retries |
| Note has all five section headings | `write_file` rejected, missing sections named |
| Notes can only be replaced whole | `edit_file` and `delete` disabled |
| Reply to parent ≤ 500 words (`RESULT_MAX_WORDS`) | final message truncated after the run; path line comes first so it survives |
| Reply starts with `Notes file: <path>` | set in code from the `write_file` that actually succeeded (`NOT SAVED` if none); a model-written path is replaced |
| Final report ≤ 700 words (`REPORT_MAX_WORDS`) | constant defined; **[planned]** applied when the report writer exists |

Content rules that are prompt-level only (no practical code check): no raw page dumps, no long
quotations, never invent figures, flag unverified/stale claims. Every figure should keep its
source URL and date.

## 7. Orchestrator [planned]

`create_deep_agent` with the same sandboxed backend and `subagents=[get_researcher_subagent()]`.
Its prompt will: plan with `write_todos`; split the request into at most 3 focused briefs
(e.g. financial performance, risks, outlook, each covering both companies); delegate with
`task`; collect note paths; read the notes; write `/report.md`. Model: `ORCHESTRATOR_MODEL`.
`main.py` will call `start_request()` and then invoke it.

## 8. Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `TAVILY_API_KEY` | Secrets, loaded via `python-dotenv`; only presence is ever checked |
| `ORCHESTRATOR_MODEL`, `RESEARCHER_MODEL` | Model names, default `gpt-6.1-terra` |
| `TAVILY_MAX_CALLS_PER_TASK` / `_MAX_RESEARCH_TASKS` / `_MAX_CALLS_PER_REQUEST` | Budget, clamped to 5 / 3 / 15 |

`.env` is git-ignored; `.env.example` lists names only.

## 9. Testing

Offline and light, run with `python -m pytest -q`:

- `test_tavily_search.py`: budget, gap reason, cache, ordering, logging, no key leak (fake search).
- `test_think_tool.py`: reflection checks.
- `test_researcher.py`: guard rules, plus the researcher running through the parent's `task`
  tool with scripted fake models: execution, artifact creation, ≤ note limit, ≤ 500-word reply,
  reuse across two briefs.
- `test_request.py`, `test_structure.py`: request artifact (latest copy, archive history) and project layout.

Live checks cost API calls: `python -m fin_agent.tools.check_tavily` and
`python -m fin_agent.agents.researcher "<brief>"`.

## 10. Design decisions and limits

- Deep Agents' built-in planning, `task` delegation and filesystem tools are the core; no hand-built graph.
- One reusable subagent definition, not one agent per research category.
- Budgets and size limits are enforced in code, not trusted to prompts.
- `task_id` is model-chosen, so a model could invent a new id per search; the 15-call request cap
  still bounds total spend.
- Truncating the reply is a blunt backstop; the prompt asks for under 250 words.
- Out of scope: vector DBs, RAG, FastAPI/UI, Docker/Kubernetes, databases.
