import builtins
from contextlib import nullcontext
import errno
import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bioptim.interfaces import compiled_thread_map_external as compiled


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("initial_contents", [b"", b"existing"])
def test_windows_artifact_lock_without_fcntl_retries_and_releases(tmp_path, monkeypatch, body_fails, initial_contents):
    calls = []
    sleeps = []
    real_import = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'")
        return real_import(name, *args, **kwargs)

    def locking(fd, mode, size):
        calls.append((mode, size, os.lseek(fd, 0, os.SEEK_CUR)))
        if mode == "lock" and len(calls) <= 120:
            raise OSError(errno.EACCES, "already locked")

    monkeypatch.setattr(builtins, "__import__", without_fcntl)
    spec = importlib.util.spec_from_file_location("compiled_without_fcntl", compiled.__file__)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    monkeypatch.setattr(isolated, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(isolated, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK="lock", LK_UNLCK="unlock"))
    path = tmp_path / "nested" / "artifact.lock"
    path.parent.mkdir()
    path.write_bytes(initial_contents)

    def run():
        with isolated._artifact_lock(path):
            assert calls[-1] == ("lock", 1, 0)
            if body_fails:
                raise ValueError("cache fill failed")

    if body_fails:
        with pytest.raises(ValueError, match="cache fill failed"):
            run()
    else:
        run()
    assert sleeps == [0.1] * 120
    assert calls == [("lock", 1, 0)] * 121 + [("unlock", 1, 0)]
    assert path.read_bytes() == initial_contents


@pytest.mark.parametrize("error_number", [errno.EBADF, errno.EINVAL, errno.EPERM])
def test_windows_artifact_lock_propagates_acquisition_errors(tmp_path, monkeypatch, error_number):
    calls = []
    failure = OSError(error_number, "lock failed")

    def locking(fd, mode, size):
        calls.append(mode)
        raise failure

    monkeypatch.setattr(compiled, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK="lock", LK_UNLCK="unlock"))
    with pytest.raises(OSError) as caught:
        with compiled._artifact_lock(tmp_path / "artifact.lock"):
            pytest.fail("cache fill must not start without the lock")
    assert caught.value is failure
    assert calls == ["lock"]


def test_windows_artifact_lock_propagates_release_errors(tmp_path, monkeypatch):
    failure = OSError(errno.EIO, "unlock failed")

    def locking(fd, mode, size):
        if mode == "unlock":
            raise failure

    monkeypatch.setattr(compiled, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK="lock", LK_UNLCK="unlock"))
    with pytest.raises(OSError) as caught:
        with compiled._artifact_lock(tmp_path / "artifact.lock"):
            pass
    assert caught.value is failure


@pytest.mark.parametrize("body_fails", [False, True])
def test_artifact_lock_serializes_processes_and_releases(tmp_path, body_fails):
    """Exercise the real host lock, including release after a failed fill."""

    path = tmp_path / "nested" / "artifact.lock"
    script = """
import runpy
import sys
from pathlib import Path
lock = runpy.run_path(sys.argv[1])["_artifact_lock"]
print("ready", flush=True)
with lock(Path(sys.argv[2])):
    print("acquired", flush=True)
"""
    child = None
    try:
        with pytest.raises(ValueError, match="cache fill failed") if body_fails else nullcontext():
            with compiled._artifact_lock(path):
                child = subprocess.Popen(
                    [sys.executable, "-c", script, compiled.__file__, str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert child.stdout.readline().strip() == "ready"
                with pytest.raises(subprocess.TimeoutExpired):
                    child.communicate(timeout=0.2)
                if body_fails:
                    raise ValueError("cache fill failed")
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert stdout.strip() == "acquired"
        assert path.is_file()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()
