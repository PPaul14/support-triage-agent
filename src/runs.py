"""Helpers the runners share: the golden cases in pool order, run ids, JSONL records and progress lines."""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.label import CLEAN_JSONL, POOL_JSONL, read_jsonl

TRACES_JSONL = Path(__file__).resolve().parent.parent / "artifacts" / "traces.jsonl"  # quotes tweets: gitignored


def golden_cases(limit: int | None = None) -> list[dict]:
    """The golden cases in pool order (a seeded shuffle), or only the first `limit` of them."""
    case_ids = [entry["case_id"] for entry in read_jsonl(POOL_JSONL)]
    if limit is not None:
        case_ids = case_ids[:limit]
    wanted = set(case_ids)
    by_id = {case["case_id"]: case for case in read_jsonl(CLEAN_JSONL) if case["case_id"] in wanted}
    return [by_id[case_id] for case_id in case_ids]


def new_run_id() -> str:
    """The run's UTC start time, so run ids sort in time order."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_trace(record: object, path: Path = TRACES_JSONL) -> None:
    """Append one dataclass record to a JSONL file (the traces by default) as one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def progress(stage: str, number: int, total: int, started: float) -> None:
    """One line per finished item, for tailing the run log. The time left is a straight-line guess."""
    elapsed = time.perf_counter() - started
    left = elapsed / number * (total - number)
    print(f"{datetime.now():%H:%M:%S} {stage} {number}/{total} | {elapsed / 60:.1f} min elapsed, "
          f"about {left / 60:.1f} min left", flush=True)
