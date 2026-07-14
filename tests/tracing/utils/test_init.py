import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlflow.tracing.utils import map_with_context

_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_ctx_var", default="unset")


def test_map_with_context_propagates_contextvars_to_workers():
    _ctx_var.set("caller-value")

    def _read(_):
        return _ctx_var.get()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, _read, range(8)))

    assert results == ["caller-value"] * 8


def test_map_with_context_is_safe_under_concurrency():
    # More items than workers, with overlap, so tasks run concurrently. A single
    # shared context copy would raise "context is already entered" here.
    _ctx_var.set("caller-value")

    def _read(_):
        time.sleep(0.02)
        return _ctx_var.get()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, _read, range(16)))

    assert results == ["caller-value"] * 16


def test_map_with_context_preserves_order():
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, lambda x: x * x, range(6)))

    assert results == [0, 1, 4, 9, 16, 25]


def test_map_with_context_does_not_leak_worker_mutations_to_caller():
    _ctx_var.set("caller-value")

    def _mutate(x):
        _ctx_var.set(f"worker-{x}")
        return _ctx_var.get()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, _mutate, range(4)))

    assert sorted(results) == ["worker-0", "worker-1", "worker-2", "worker-3"]
    assert _ctx_var.get() == "caller-value"


def test_map_with_context_propagates_exceptions():
    def _boom(x):
        if x == 2:
            raise ValueError("boom")
        return x

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ValueError, match="boom"):
            list(map_with_context(executor, _boom, range(4)))


def test_map_with_context_handles_empty_iterable():
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(map_with_context(executor, lambda x: x, [])) == []
