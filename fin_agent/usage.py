"""Count LLM calls and tokens per agent for one run (a LangChain callback; no extra services).

Passed in `config["callbacks"]`, it also sees the researcher subagent's calls. It records a
call when it STARTS and again when it ends, so a request that never returns (hung) shows up
as started-but-not-finished instead of being invisible.
"""
from __future__ import annotations

import threading
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler


class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._who: dict = {}  # run_id -> agent label, for calls in flight
        self.stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"started": 0, "finished": 0, "input": 0, "output": 0, "cached": 0}
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        who = (metadata or {}).get("lc_agent_name") or "unknown"
        with self._lock:
            self._who[run_id] = who
            self.stats[who]["started"] += 1

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage: dict = {}
        try:
            usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
        except (IndexError, AttributeError):
            pass
        with self._lock:
            s = self.stats[self._who.pop(run_id, "unknown")]
            s["finished"] += 1
            s["input"] += int(usage.get("input_tokens", 0))
            s["output"] += int(usage.get("output_tokens", 0))
            s["cached"] += int((usage.get("input_token_details") or {}).get("cache_read", 0))

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            self._who.pop(run_id, None)

    def summary(self) -> str:
        with self._lock:
            rows = dict(self.stats)
        if not rows:
            return "LLM usage: no calls recorded."
        head = f"{'agent':<14}{'calls':>6}{'unfinished':>11}{'input tok':>11}{'output tok':>11}{'cached in':>11}"
        lines = ["LLM usage for this request:", head]
        total = {"started": 0, "finished": 0, "input": 0, "output": 0, "cached": 0}
        for who, s in sorted(rows.items()):
            lines.append(
                f"{who:<14}{s['started']:>6}{s['started'] - s['finished']:>11}"
                f"{s['input']:>11}{s['output']:>11}{s['cached']:>11}"
            )
            for k in total:
                total[k] += s[k]
        lines.append(
            f"{'TOTAL':<14}{total['started']:>6}{total['started'] - total['finished']:>11}"
            f"{total['input']:>11}{total['output']:>11}{total['cached']:>11}"
        )
        return "\n".join(lines)
