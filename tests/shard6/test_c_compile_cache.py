import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from bioptim import Solver
from bioptim.interfaces.c_compile_cache import cached_nlpsol


OPTIONS = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}
LIMITS = {"x0": [0.8, 0.8], "lbg": 1, "ubg": 1}


def _nlp(offset=1.0):
    x = ca.SX.sym("x", 2)
    return {"x": x, "f": (x[0] - offset) ** 2 + 3 * (x[1] - 2) ** 2, "g": x[0] + x[1]}


def test_ipopt_compilation_options(tmp_path):
    options = Solver.IPOPT()
    options.set_c_compile(True, compiler_flags=["-O1"], cache_dir=tmp_path, cache_name="rho")
    assert options.c_compile
    assert options.compiler_flags == ("-O1",)
    assert options.cache_dir == str(tmp_path)
    assert options.cache_name == "rho"
    values = options.as_dict(type("Interface", (), {"options_common": {}})())
    assert not any(any(name in key for name in ("compile", "cache")) for key in values)
    options.set_c_compile(True)
    assert options.compiler_flags == () and options.cache_dir is None and options.cache_name == "ipopt"
    with pytest.raises(TypeError, match="compiler_flags"):
        options.set_c_compile(True, compiler_flags="-O1")
    with pytest.raises(ValueError, match="cache_name"):
        options.set_c_compile(True, cache_name="../unsafe")


@pytest.mark.skipif(platform.system() != "Linux" or not shutil.which("gcc"), reason="Linux/GCC cache")
def test_sx_cache_cold_warm_second_process_and_invalidation(tmp_path):
    vm = ca.nlpsol("reference", "ipopt", _nlp(), OPTIONS)
    reference = vm(**LIMITS)
    native, cold = cached_nlpsol(
        "ipopt", _nlp(), OPTIONS, compiler_flags=("-O1",), cache_dir=tmp_path, cache_name="test"
    )
    assert not cold["hit"]
    result = native(**LIMITS)
    for key in ("x", "f", "g", "lam_x", "lam_g"):
        np.testing.assert_allclose(result[key], reference[key], rtol=1e-11, atol=1e-11)
    assert native.stats()["iter_count"] == vm.stats()["iter_count"]
    # A fresh interpreter must reuse the .so; prohibit compilation rather than
    # using a noisy wall-clock threshold to infer that a hit happened.
    script = """
import json, sys
import casadi as ca
import bioptim.interfaces.c_compile_cache as cache
def forbid_compile(*args, **kwargs):
    raise AssertionError('warm process attempted compilation')
# check_output uses run internally too, so preserve version probes.
original_run = cache.subprocess.run
def guarded_run(command, *args, **kwargs):
    if '-shared' in command:
        return forbid_compile()
    return original_run(command, *args, **kwargs)
cache.subprocess.run = guarded_run
x = ca.SX.sym('x', 2)
nlp = {'x': x, 'f': (x[0] - 1.0)**2 + 3*(x[1] - 2)**2, 'g': x[0] + x[1]}
solver, info = cache.cached_nlpsol('ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-10},
    compiler_flags=('-O1',), cache_dir=sys.argv[1], cache_name='test')
result = solver(x0=[0.8, 0.8], lbg=1, ubg=1)
print(json.dumps({'info': info, 'x': result['x'].full().ravel().tolist(), 'iterations': solver.stats()['iter_count']}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=environment, check=True
    )
    warm = json.loads(process.stdout.splitlines()[-1])
    assert warm["info"]["hit"] and warm["info"]["key"] == cold["key"]
    assert warm["iterations"] == vm.stats()["iter_count"]
    np.testing.assert_allclose(warm["x"], reference["x"].full().ravel(), atol=1e-11)
    _, changed_flags = cached_nlpsol(
        "ipopt", _nlp(), OPTIONS, compiler_flags=("-O0",), cache_dir=tmp_path, cache_name="test"
    )
    _, changed_graph = cached_nlpsol(
        "ipopt", _nlp(1.5), OPTIONS, compiler_flags=("-O1",), cache_dir=tmp_path, cache_name="test"
    )
    assert not changed_flags["hit"] and not changed_graph["hit"]
    assert len({cold["key"], changed_flags["key"], changed_graph["key"]}) == 3
    manifest = Path(cold["library"]).with_name("manifest.json")
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="Invalid compiled NLP cache entry"):
        cached_nlpsol("ipopt", _nlp(), OPTIONS, compiler_flags=("-O1",), cache_dir=tmp_path, cache_name="test")


@pytest.mark.skipif(platform.system() != "Linux" or not shutil.which("gcc"), reason="Linux/GCC cache")
def test_cache_failed_compilation_is_not_published(tmp_path, monkeypatch):
    import bioptim.interfaces.c_compile_cache as cache

    original_run = subprocess.run

    def fail_compile(command, *args, **kwargs):
        if "-shared" in command:
            return subprocess.CompletedProcess(command, 1, "", "test compiler failure")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(cache.subprocess, "run", fail_compile)
    with pytest.raises(RuntimeError, match="test compiler failure"):
        cached_nlpsol("ipopt", _nlp(), OPTIONS, compiler_flags=("-O1",), cache_dir=tmp_path, cache_name="test")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(platform.system() != "Linux" or not shutil.which("gcc"), reason="Linux/GCC cache")
def test_generic_solve_compilation_configuration_rebuilds_solver(tmp_path, monkeypatch):
    from bioptim.interfaces import interface_utils
    from bioptim.interfaces.solver_interface import SolverInterface

    nlp = _nlp()
    ocp = SimpleNamespace(
        variables_vector=nlp["x"],
        bounds_vectors=(np.full((2, 1), -np.inf), np.full((2, 1), np.inf)),
        init_vector=np.array([[0.8], [0.8]]),
    )
    interface = SolverInterface(ocp)
    interface.opts = Solver.IPOPT()
    interface.opts.set_print_level(0)
    interface.options_common = {"print_time": False}
    interface.solver_name = "IPOPT"
    interface.lam_x = interface.lam_g = None
    interface.shaked_objectives = nlp["f"]
    interface.shaked_constraints = nlp["g"]
    interface.dispatch_obj_func = lambda: nlp["f"]
    interface.dispatch_bounds = lambda: (nlp["g"], SimpleNamespace(min=1, max=1))
    monkeypatch.setattr(interface_utils, "_vectors_are_equal", lambda *args, **kwargs: True)

    reference = interface_utils.generic_solve(interface, False)["sol"]["x"]
    interface.opts.set_c_compile(True, compiler_flags=["-O1"], cache_dir=tmp_path / "cache")
    native = interface_utils.generic_solve(interface, False)["sol"]["x"]
    np.testing.assert_allclose(native, reference, atol=1e-11)
    assert interface.c_compile and not interface.c_compile_cache_info["hit"]
    first_solver = interface.shaked_ocp_solver
    interface_utils.generic_solve(interface, False)
    assert interface.shaked_ocp_solver is first_solver
    interface.opts.set_c_compile(True, compiler_flags=["-O0"], cache_dir=tmp_path / "cache")
    interface_utils.generic_solve(interface, False)
    assert interface.shaked_ocp_solver is not first_solver
    interface.opts.set_c_compile(False)
    interface_utils.generic_solve(interface, False)
    assert not interface.c_compile and interface.c_compile_cache_info is None
    # The historical nonpersistent route also accepts optimizer flags.
    monkeypatch.chdir(tmp_path)
    interface.opts.set_c_compile(True, compiler_flags=["-O1"])
    native = interface_utils.generic_solve(interface, False)["sol"]["x"]
    np.testing.assert_allclose(native, reference, atol=1e-11)
    assert interface.c_compile and interface.c_compile_cache_info is None
