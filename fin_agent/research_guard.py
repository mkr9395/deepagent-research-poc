"""Application-level enforcement of the research output limits.

Prompts ask for short output; this middleware makes it a guarantee:

* The researcher may only write `/notes/<slug>.md` (virtual path inside the
  sandboxed `outputs/research/` backend). `edit_file` / `delete` are refused so
  a note can only be replaced as a whole, through the validated path.
* A note must be <= NOTE_MAX_WORDS and contain the five required sections.
  Violations are returned to the model as an error so it can shorten and retry.
* The final message handed back to the parent is hard-truncated to
  RESULT_MAX_WORDS (the note path is requested first, so it survives).
"""
from __future__ import annotations

import logging
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger("fin_agent.research_guard")

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

    def after_agent(self, state, runtime):
        """Hard-cap the final message returned to the parent agent."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.text:
                words = count_words(msg.text)
                if words <= RESULT_MAX_WORDS:
                    return None
                logger.warning("Researcher result was %d words; truncating to %d", words, RESULT_MAX_WORDS)
                # Same id => the add_messages reducer replaces the original message.
                return {"messages": [AIMessage(content=truncate_words(msg.text, RESULT_MAX_WORDS), id=msg.id)]}
        return None
