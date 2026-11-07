"""Offline tests: scripted fake models + fake Tavily search. No network, no API keys."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from fin_agent.agents.researcher import get_researcher_subagent
from fin_agent.research_guard import (
    NOTE_MAX_WORDS,
    RESULT_MAX_WORDS,
    count_words,
    truncate_words,
    validate_write,
)
from fin_agent.tools import tavily_search as ts
from fin_agent.tools.tavily_search import Limits, TavilyGuard

SECTIONS = "## Research Question\nq\n## Key Findings\nf\n## Financial Facts\nx\n## Sources\nhttps://sec.gov/a\n## Caveats\nc\n"
GOOD_NOTE = "# Note\n" + SECTIONS
PATH = "/notes/jpm-bac-revenue.md"


class ScriptedModel(BaseChatModel):
    script: list
    idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[self.idx]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{name}{len(str(args))}"}])


def test_validate_write():
    assert validate_write(PATH, GOOD_NOTE) is None
    assert "must be saved as" in validate_write("/report.md", GOOD_NOTE)
    assert "must be saved as" in validate_write("/notes/../x.md", GOOD_NOTE)
    assert "must be saved as" in validate_write("/notes/a/b.md", GOOD_NOTE)
    too_long = GOOD_NOTE + "word " * NOTE_MAX_WORDS
    assert "limit is 350" in validate_write(PATH, too_long)
    assert "Caveats" in validate_write(PATH, GOOD_NOTE.replace("## Caveats", "## Other"))


def test_truncate_words():
    text = "alpha beta\ngamma delta epsilon"
    assert truncate_words(text, 10) == text
    out = truncate_words(text, 3)
    assert out.startswith("alpha beta\ngamma") and "delta" not in out


def test_researcher_via_task_enforces_limits(tmp_path, monkeypatch, caplog):
    class FakeSearch:
        calls = []

        def __call__(self, q):
            self.calls.append(q)
            return {"query": q, "results": [{"title": "10-K", "url": "https://sec.gov/a", "content": "Revenue up"}]}

    fake = FakeSearch()
    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=fake))

    too_long = GOOD_NOTE + "word " * 400
    researcher_model = ScriptedModel(script=[
        call("tavily_search", query="JPM BAC revenue 2025", task_id="jpm-bac-revenue"),
        call("think_tool", evidence_summary="found", evidence_sufficient=True),
        call("write_file", file_path=PATH, content=too_long),   # rejected by code
        call("edit_file", file_path=PATH, old_string="a", new_string="b"),  # disabled
        call("write_file", file_path=PATH, content=GOOD_NOTE),  # accepted
        AIMessage(content=f"Notes file: {PATH}\n" + "finding " * 700),  # over the limit
    ])
    orchestrator_model = ScriptedModel(script=[
        call("task", description="Research revenue for JPM and BAC", subagent_type="researcher"),
        AIMessage(content="done"),
    ])
    spec = get_researcher_subagent(model=researcher_model)
    agent = create_deep_agent(
        model=orchestrator_model,
        subagents=[spec],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    with caplog.at_level("WARNING", logger="fin_agent.research_guard"):
        out = agent.invoke({"messages": [{"role": "user", "content": "compare"}]})
    assert "Blocked write_file: Error: note is" in caplog.text  # long note rejected
    assert "Blocked edit_file" in caplog.text
    assert "truncating to 500" in caplog.text

    # the note was written inside the sandbox root, within the limit
    note = tmp_path / "notes" / "jpm-bac-revenue.md"
    assert note.exists()
    assert count_words(note.read_text(encoding="utf-8")) <= NOTE_MAX_WORDS
    assert note.read_text(encoding="utf-8") == GOOD_NOTE

    # the guard rejected the long write and the edit before the good write
    sub_msgs = [m for m in out["messages"] if getattr(m, "name", "") == "task"]
    assert sub_msgs, "task tool result missing"
    result = sub_msgs[-1].content
    assert result.startswith(f"Notes file: {PATH}")
    assert count_words(result) <= RESULT_MAX_WORDS + len("[truncated to fit the word limit]".split())
    assert len(fake.calls) == 1
    assert researcher_model.idx == 6  # every scripted step ran, i.e. errors let the model retry
