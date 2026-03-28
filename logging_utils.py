"""Helpers for teeing pipeline output to a log file."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import TextIO

DEFAULT_LOG_DIR = Path("logs")


class TeeStream:
    """Write the same text to multiple streams."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_output(log_path: str | None, latest_alias: str | None = None):
    """Duplicate stdout/stderr to a log file when a path is provided."""
    if not log_path:
        yield
        return

    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if latest_alias:
        alias = Path(latest_alias).expanduser()
        alias.parent.mkdir(parents=True, exist_ok=True)
        try:
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            alias.symlink_to(path.resolve())
        except OSError:
            alias.write_text(f"{path}\n", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with path.open("a", encoding="utf-8") as log_handle:
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)
        print(f"[quant-log] {datetime.now().isoformat()} logging to {path}")
        try:
            yield
        finally:
            print(f"[quant-log] {datetime.now().isoformat()} run finished")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def default_log_path(log_prefix: str) -> str:
    """Return a timestamped default log path for quant-pipeline scripts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(DEFAULT_LOG_DIR / f"{log_prefix}_{timestamp}.log")


def latest_log_alias(log_prefix: str) -> str:
    """Return the stable alias path for the latest quant-pipeline script log."""
    return str(DEFAULT_LOG_DIR / f"latest_{log_prefix}.log")


@contextmanager
def tee_output_from_env(log_prefix: str | None = None):
    """Enable tee logging via QUANT_LOG_FILE or a default quant-pipeline log file."""
    log_path = os.getenv("QUANT_LOG_FILE")
    latest_alias = latest_log_alias(log_prefix) if log_prefix else None
    if not log_path and log_prefix:
        log_path = default_log_path(log_prefix)
    with tee_output(log_path, latest_alias=latest_alias):
        yield
