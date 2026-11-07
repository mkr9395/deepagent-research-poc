"""Preserve the user's original request as markdown artifacts.

Each request is written twice: `original_request.md` always holds the latest one,
and a timestamped copy goes to `requests/` next to it so history is kept.
"""
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path("outputs") / "original_request.md"


def save_original_request(
    query: str,
    path: Path = DEFAULT_PATH,
    companies: list[str] | None = None,
    dimensions: list[str] | None = None,
    archive_dir: Path | None = None,
) -> Path:
    """Write the request (and optional normalized scope) to `path` (latest request,
    overwritten) and to a new timestamped file in `archive_dir` (default: `requests/`
    beside `path`). Returns `path`. Archive copies are never overwritten."""
    query = query.strip()
    if not query:
        raise ValueError("Request is empty")
    now = datetime.now()

    lines = [
        "# Original Request",
        "",
        f"_Saved: {now.isoformat(timespec='seconds')}_",
        "",
        "## User question",
        "",
        query,
    ]
    if companies or dimensions:
        lines += ["", "## Normalized scope", ""]
        if companies:
            lines.append(f"- Companies: {', '.join(companies)}")
        if dimensions:
            lines.append(f"- Dimensions: {', '.join(dimensions)}")

    text = "\n".join(lines) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    archive_dir = Path(archive_dir) if archive_dir else path.parent / "requests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    archive = archive_dir / f"{stamp}.md"
    n = 1
    while archive.exists():  # two requests in the same second: never overwrite
        n += 1
        archive = archive_dir / f"{stamp}_{n}.md"
    archive.write_text(text, encoding="utf-8")
    return path
