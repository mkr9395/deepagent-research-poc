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
    assert "limit is 500" in validate_write(PATH, too_long)
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

    too_long = GOOD_NOTE + "word " * (NOTE_MAX_WORDS + 50)
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


# ---- focused tests (one behaviour each) -------------------------------------
def run_researcher(tmp_path, monkeypatch, write_steps, final_text, briefs=1):
    """Run the researcher through the parent's `task` tool with scripted models.

    `write_steps`: list (per brief) of lists of (path, content) write_file attempts.
    Returns (parent_output, task_results, researcher_models).
    """
    def fake_search(q):
        return {"query": q, "results": [{"title": "10-K", "url": "https://sec.gov/a", "content": "Revenue up"}]}

    monkeypatch.setattr(ts, "_guard", TavilyGuard(Limits(), search_fn=fake_search))
    researcher_scripts, parent_script = [], []
    for i in range(briefs):
        researcher_scripts.append([
            call("tavily_search", query=f"query {i}", task_id=f"task-{i}"),
            call("think_tool", evidence_summary="found", evidence_sufficient=True),
            *[call("write_file", file_path=p, content=c) for p, c in write_steps[i]],
            AIMessage(content=final_text),
        ])
        parent_script.append(call("task", description=f"brief {i}", subagent_type="researcher"))
    parent_script.append(AIMessage(content="done"))

    # one researcher definition, a fresh scripted model per invocation
    models = [ScriptedModel(script=s) for s in researcher_scripts]
    spec = get_researcher_subagent(model=models[0])
    if briefs > 1:  # reuse the same definition: swap in the next scripted model per call
        it = iter(models)
        spec["model"] = _SequencedModel(models=models, it=it)
    agent = create_deep_agent(
        model=ScriptedModel(script=parent_script),
        subagents=[spec],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "compare"}]})
    results = [m.content for m in out["messages"] if getattr(m, "name", "") == "task"]
    return out, results, models


class _SequencedModel(ScriptedModel):
    """Delegates each new subagent run (detected by an exhausted script) to the next model."""
    models: list
    it: object = None
    script: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        cur = next((m for m in self.models if m.idx < len(m.script)), self.models[-1])
        return cur._generate(messages, stop, run_manager, **kwargs)


def test_subagent_executes_and_returns_note_path(tmp_path, monkeypatch):
    _, results, models = run_researcher(
        tmp_path, monkeypatch, [[(PATH, GOOD_NOTE)]], f"Notes file: {PATH}\n- Revenue up 5% (sec.gov)"
    )
    assert models[0].idx == len(models[0].script)  # ran search -> think -> write -> reply
    assert len(results) == 1 and PATH in results[0]


def test_research_artifact_is_created_with_required_sections(tmp_path, monkeypatch):
    run_researcher(tmp_path, monkeypatch, [[(PATH, GOOD_NOTE)]], f"Notes file: {PATH}")
    note = tmp_path / "notes" / "jpm-bac-revenue.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "https://" in text
    assert missing_sections_of(text) == []


def missing_sections_of(text):
    from fin_agent.research_guard import missing_sections
    return missing_sections(text)


def test_oversized_artifact_is_never_saved(tmp_path, monkeypatch):
    too_long = GOOD_NOTE + "word " * (NOTE_MAX_WORDS + 50)
    run_researcher(tmp_path, monkeypatch, [[(PATH, too_long)]], "gave up")
    assert not (tmp_path / "notes" / "jpm-bac-revenue.md").exists()


def test_artifact_at_limit_is_saved_and_within_note_limit(tmp_path, monkeypatch):
    pad = NOTE_MAX_WORDS - count_words(GOOD_NOTE)
    note = GOOD_NOTE + "w " * pad
    assert count_words(note) == NOTE_MAX_WORDS
    run_researcher(tmp_path, monkeypatch, [[(PATH, note)]], f"Notes file: {PATH}")
    saved = (tmp_path / "notes" / "jpm-bac-revenue.md").read_text(encoding="utf-8")
    assert count_words(saved) <= NOTE_MAX_WORDS


def test_returned_result_is_capped_at_500_words(tmp_path, monkeypatch):
    _, results, _ = run_researcher(
        tmp_path, monkeypatch, [[(PATH, GOOD_NOTE)]], f"Notes file: {PATH}\n" + "finding " * 900
    )
    assert results[0].startswith(f"Notes file: {PATH}")  # path survives truncation
    body = results[0].replace("[truncated to fit the word limit]", "")
    assert count_words(body) <= RESULT_MAX_WORDS


def test_short_result_is_left_untouched(tmp_path, monkeypatch):
    reply = f"Notes file: {PATH}\n- Revenue up 5%"
    _, results, _ = run_researcher(tmp_path, monkeypatch, [[(PATH, GOOD_NOTE)]], reply)
    assert results[0] == reply


def test_same_researcher_is_reusable_for_two_briefs(tmp_path, monkeypatch):
    p2 = "/notes/jpm-bac-risk.md"
    _, results, _ = run_researcher(
        tmp_path, monkeypatch, [[(PATH, GOOD_NOTE)], [(p2, GOOD_NOTE)]], "Notes file: see note", briefs=2
    )
    assert len(results) == 2
    assert (tmp_path / "notes" / "jpm-bac-revenue.md").exists()
    assert (tmp_path / "notes" / "jpm-bac-risk.md").exists()
