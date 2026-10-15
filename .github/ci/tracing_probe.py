"""Record tracing left enabled at test boundaries without changing test outcomes."""

import json
import os
from pathlib import Path
import tracemalloc

import pytest

_leaks = []


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    if tracemalloc.is_tracing():
        _leaks.append(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    Path(os.environ["TRACING_AUDIT_FILE"]).write_text(json.dumps(_leaks, indent=2))
