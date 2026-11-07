"""Offline checks of the overall run limit and the per-request model timeout."""
import time

import main
from fin_agent.models import build_model, request_timeout_seconds


class SlowAgent:
    def invoke(self, *a, **kw):
        time.sleep(5)
        return {"messages": []}


def test_overall_run_limit_stops_a_hung_run(monkeypatch, capsys):
    monkeypatch.setenv("RUN_TIME_LIMIT_SECONDS", "0.5")
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(main, "save_original_request", lambda q: "x.md")
    monkeypatch.setattr(main, "get_backend", lambda: None)
    monkeypatch.setattr(main, "reset_run_artifacts", lambda: None)
    monkeypatch.setattr(main, "start_request", lambda: None)
    monkeypatch.setattr(main, "build_orchestrator", lambda: SlowAgent())
    t0 = time.monotonic()
    code = main.run("compare A and B")
    assert code == 1 and time.monotonic() - t0 < 3
    out = capsys.readouterr().out
    assert "RUN STOPPED" in out and "LLM usage" in out


def test_run_limit_default_and_disable(monkeypatch):
    monkeypatch.delenv("RUN_TIME_LIMIT_SECONDS", raising=False)
    assert main.run_time_limit_seconds() == 480
    monkeypatch.setenv("RUN_TIME_LIMIT_SECONDS", "0")
    assert main.run_time_limit_seconds() == 0
    monkeypatch.setenv("RUN_TIME_LIMIT_SECONDS", "junk")
    assert main.run_time_limit_seconds() == 480


def test_models_get_request_timeout_and_one_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    monkeypatch.delenv("OPENAI_REQUEST_TIMEOUT_SECONDS", raising=False)
    m = build_model("openai:gpt-test")
    assert m.request_timeout == 90 and m.max_retries == 1
    monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "30")
    assert request_timeout_seconds() == 30
