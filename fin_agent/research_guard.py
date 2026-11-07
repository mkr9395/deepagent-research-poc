"""Application-level enforcement of the research output limits.

Prompts ask for short output; this middleware makes it a guarantee:

* The researcher may only write `/notes/<slug>.md` (virtual path inside the
  sandboxed `outputs/research/` backend). `edit_file` / `delete` are refused so
  a note can only be replaced as a whole, through the validated path.
* A note must be <= NOTE_MAX_WORDS and contain the five required sections.
  Violations are returned to the model as an error so it can shorten and retry.
* The final message handed back to the parent is normalised in code: its first line is
  always `Notes file: <path>` of the note actually saved (or `NOT SAVED`), whatever the
  model wrote, and the whole message is hard-truncated to RESULT_MAX_WORDS.
  The parent therefore always gets a concise result plus a verified artifact reference.
"""
from __future__ import annotations

import logging
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger("fin_agent.research_guard")
# What crosses the parent <-> researcher boundary (one line each way). Enable with INFO.
ctx_log = logging.getLogger("fin_agent.context")

NOTE_MAX_WORDS = 500
RESULT_MAX_WORDS = 500
REPORT_MAX_WORDS = 700  # final report (applied when the report writer is built)
NOTES_DIR = "/notes"
NOTE_PATH_RE = re.compile(rf"^{NOTES_DIR}/[a-z0-9][a-z0-9_-]{{0,80}}\.md$")
REQUIRED_SECTIONS = [
    "research question",
    "key findings",
    "financial facts",
    "sources",
    "caveats",
]
_TRUNCATION_MARK = "\n[truncated to fit the word limit]"


def count_words(text: str) -> int:
    return len(text.split())


def truncate_words(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` words (keeps line breaks of the kept part)."""
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= limit:
        return text
    return text[: matches[limit - 1].end()] + _TRUNCATION_MARK


NOTES_FILE_PREFIX = "Notes file:"


def last_saved_note(messages) -> str | None:
    """Path of the last note this run actually saved (write_file that succeeded)."""
    written = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == "write_file":
                written[tc["id"]] = (tc.get("args") or {}).get("file_path", "").strip()
    saved = [
        written[m.tool_call_id]
        for m in messages
        if isinstance(m, ToolMessage) and m.tool_call_id in written and m.status != "error"
    ]
    return saved[-1] if saved else None


def with_note_reference(text: str, path: str | None) -> str:
    """Make the first line `Notes file: <path>`, dropping any model-written variant."""
    body = "\n".join(l for l in text.splitlines() if not l.strip().lower().startswith(NOTES_FILE_PREFIX.lower()))
    head = (
        f"{NOTES_FILE_PREFIX} {path}"
        if path
        else f"{NOTES_FILE_PREFIX} NOT SAVED (no research note was written; treat the findings below as unverified)"
    )
    return f"{head}\n{body.strip()}"


def missing_sections(markdown: str) -> list[str]:
    headings = {
        re.sub(r"[^a-z ]", "", line.lstrip("#").strip().lower()).strip()
        for line in markdown.splitlines()
        if line.lstrip().startswith("#")
    }
    return [s for s in REQUIRED_SECTIONS if not any(h.startswith(s) for h in headings)]


def validate_write(file_path: str, content: str) -> str | None:
    """Return an error message if this write is not allowed, else None."""
    path = (file_path or "").strip()
    if not NOTE_PATH_RE.match(path):
        return (
            f"Error: refusing to write {file_path!r}. Research notes must be saved as "
            f"{NOTES_DIR}/<task-slug>.md (lowercase letters, digits, - or _)."
        )
    words = count_words(content)
    if words > NOTE_MAX_WORDS:
        return (
            f"Error: note is {words} words; the limit is {NOTE_MAX_WORDS}. Nothing was saved. "
            f"Cut it by at least {words - NOTE_MAX_WORDS} words (drop minor facts, shorten "
            "sentences, keep every source URL) and call write_file again with the same path."
        )
    missing = missing_sections(content)
    if missing:
        return (
            "Error: note is missing required section heading(s): "
            + ", ".join(f"'## {m.title()}'" for m in missing)
            + ". Nothing was saved. Rewrite it with all five sections."
        )
    return None


class ResearchGuardMiddleware(AgentMiddleware):
    """Enforces note path / note size on file writes and result size on exit."""

    def _check(self, request) -> ToolMessage | None:
        call = request.tool_call
        name, args = call["name"], call.get("args", {})
        error = None
        if name in ("edit_file", "delete"):
            error = (
                f"Error: {name} is disabled. To change a note, call write_file with the "
                "complete new content (it overwrites the previous version)."
            )
        elif name == "write_file":
            error = validate_write(args.get("file_path", ""), args.get("content", ""))
        if error is None:
            return None
        logger.warning("Blocked %s: %s", name, error)
        return ToolMessage(content=error, name=name, tool_call_id=call["id"], status="error")

    def wrap_tool_call(self, request, handler):
        return self._check(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self._check(request) or await handler(request)

    def before_agent(self, state, runtime):
        """Log what the parent handed to the researcher (isolated mode: just the brief)."""
        msgs = state["messages"]
        brief = next((m.text for m in msgs if m.type == "human"), "")
        ctx_log.info(
            "PARENT -> RESEARCHER | brief %d words, %d message(s) received | %r",
            count_words(brief), len(msgs), _preview(brief),
        )
        return None

    def after_agent(self, state, runtime):
        """Guarantee the note reference and the size cap on the message returned to the parent."""
        msgs = state["messages"]
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and msg.text:
                path = last_saved_note(msgs)
                text = with_note_reference(msg.text, path)
                words = count_words(text)
                if words > RESULT_MAX_WORDS:
                    logger.warning("Researcher result was %d words; truncating to %d", words, RESULT_MAX_WORDS)
                    text = truncate_words(text, RESULT_MAX_WORDS)
                _log_return(msgs, text, path)
                if text == msg.text:
                    return None
                # Same id => the add_messages reducer replaces the original message.
                return {"messages": [AIMessage(content=text, id=msg.id)]}
        return None


def _preview(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def _log_return(msgs, reply: str, note_path: str | None) -> None:
    """Log what goes back to the parent versus what stays behind in the child/on disk."""
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    searches = sum(1 for m in tool_msgs if m.name == "tavily_search")
    internal_chars = sum(len(str(m.content)) for m in tool_msgs)
    note_words = 0
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == "write_file" and (tc.get("args") or {}).get("file_path") == note_path:
                note_words = count_words(tc["args"].get("content", ""))
    ctx_log.info(
        "RESEARCHER -> PARENT | reply %d words | note=%s (%d words, stays on disk) | "
        "kept out of parent: %d messages, %d tool results (%d searches), %d chars",
        count_words(reply), note_path or "NOT SAVED", note_words,
        len(msgs), len(tool_msgs), searches, internal_chars,
    )
    ctx_log.info("RESEARCHER -> PARENT | reply text: %r", _preview(reply, 160))
