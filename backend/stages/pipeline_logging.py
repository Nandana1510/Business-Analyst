"""Terminal logging for pipeline / agent execution (no business logic)."""

from contextlib import contextmanager


@contextmanager
def agent_log(name: str):
    """Print [START] / [END] lines around one agent run."""
    print(f"[START] {name}", flush=True)
    try:
        yield
    finally:
        print(f"[END] {name}", flush=True)
