from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_project_layout():
    for p in ["main.py", "requirements.txt", "fin_agent/agents", "fin_agent/tools", "outputs/research"]:
        assert (ROOT / p).exists(), f"missing {p}"


def test_no_real_env_committed():
    assert (ROOT / ".env.example").exists()
