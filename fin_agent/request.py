"""Preserve the user's original request as a markdown artifact."""
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path("outputs") / "original_request.md"


def save_original_request(
    query: str,
    path: Path = DEFAULT_PATH,
    companies: list[str] | None = None,
    dimensions: list[str] | None = None,
) -> Path:
    """Write the request (and optional normalized scope) to `path`; overwrite any previous one."""
    query = query.strip()
    if not query:
        raise ValueError("Request is empty")

    lines = [
        "# Original Request",
        "",
        f"_Saved: {datetime.now().isoformat(timespec='seconds')}_",
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
