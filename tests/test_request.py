import pytest

from fin_agent.request import save_original_request


def test_saves_request_and_scope(tmp_path):
    out = tmp_path / "outputs" / "original_request.md"
    q = "Compare Microsoft and Alphabet on revenue growth and risks."
    save_original_request(q, out, companies=["Microsoft", "Alphabet"], dimensions=["revenue growth", "risks"])

    text = out.read_text(encoding="utf-8")
    assert q in text
    assert "Companies: Microsoft, Alphabet" in text
    assert "Dimensions: revenue growth, risks" in text


def test_overwrites_previous_request(tmp_path):
    out = tmp_path / "original_request.md"
    save_original_request("first question", out)
    save_original_request("second question", out)
    text = out.read_text(encoding="utf-8")
    assert "second question" in text and "first question" not in text


def test_empty_request_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_original_request("   ", tmp_path / "x.md")
