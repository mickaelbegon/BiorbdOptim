"""Separate native IPOPT callbacks, with a retained symbolic oracle."""
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
from bioptim.interfaces.c_compile_cache import cached_nlpsol_callbacks, IPOPT_NATIVE_CALLBACKS


OPTIONS = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}
pytestmark = pytest.mark.skipif(platform.system() != "Linux" or not shutil.which("gcc"), reason="Linux/GCC cache")


def problem(parameter=True):
    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p") if parameter else ca.SX.sym("p", 0)
    return {"x": x, "p": p, "f": (x[0] - (p if parameter else 2))**2 + (x[1] - 1)**2,
            "g": x[0] * x[1]}


def evaluate(solver, parameter=True):
    return solver(x0=[1, 1], p=2 if parameter else [], lbg=2.5, ubg=ca.inf)


def equal(a, b):
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-10)


def build(tmp_path, nlp=None, **kwargs):
    return cached_nlpsol_callbacks("ipopt", problem() if nlp is None else nlp, kwargs.pop("options", OPTIONS),
                                   compiler_flags=kwargs.pop("compiler_flags", ("-O1",)),
                                   cache_dir=tmp_path, cache_name="split", **kwargs)


@pytest.mark.parametrize("parameter", [True, False])
@pytest.mark.parametrize("transform", [True, False])
def test_separate_native_solve_and_signatures(tmp_path, parameter, transform):
    nlp = problem(parameter)
    vm = ca.nlpsol("vm", "ipopt", nlp, OPTIONS)
    prepared = vm
    if transform:
        if not hasattr(ca.Function, "transform"):
            pytest.skip("CasADi 3.8 transforms")
        from bioptim.interfaces.function_transform import transformed_nlpsol
        prepared, _ = transformed_nlpsol(nlp, OPTIONS)
    native, info = build(tmp_path, nlp, vm_solver=prepared)
    equal(evaluate(vm, parameter), evaluate(native, parameter))
    assert vm.stats()["iter_count"] == native.stats()["iter_count"]
    assert native.oracle().class_name() == "SXFunction"
    assert native.get_function("nlp_grad").class_name() == "SXFunction"
    assert info["mode"] == "callbacks" and not info["hit"]
    assert set(info["native_callbacks"]) == set(IPOPT_NATIVE_CALLBACKS)
    for name in IPOPT_NATIVE_CALLBACKS:
        original, compiled = prepared.get_function(name), native.get_function(name)
        assert compiled.class_name() == "External" and native.stats()["n_call_" + name] > 0
        assert original.name_in() == compiled.name_in() and original.name_out() == compiled.name_out()
        assert all(original.sparsity_in(i) == compiled.sparsity_in(i) for i in range(original.n_in()))
        assert all(original.sparsity_out(i) == compiled.sparsity_out(i) for i in range(original.n_out()))
        source = Path(info["callbacks"][name]["library"]).with_suffix(".c").read_text()
        assert "CASADI_SYMBOL_EXPORT int nlp(" not in source
        assert "CASADI_SYMBOL_EXPORT int nlp_grad(" not in source


def test_reload_second_process_subset_and_invalidation(tmp_path):
    solver, cold = build(tmp_path, callbacks=("nlp_g",))
    expected = evaluate(solver)
    assert solver.get_function("nlp_jac_g").class_name() == "SXFunction"
    script = """
import json, sys
import casadi as ca
import bioptim.interfaces.c_compile_cache as cache
original = cache.subprocess.run
def guarded(command, *args, **kwargs):
    if '-shared' in command:
        raise AssertionError('cache hit attempted compilation')
    return original(command, *args, **kwargs)
cache.subprocess.run = guarded
x, p = ca.SX.sym('x', 2), ca.SX.sym('p')
nlp = {'x':x, 'p':p, 'f':(x[0]-p)**2+(x[1]-1)**2, 'g':x[0]*x[1]}
solver, info = cache.cached_nlpsol_callbacks('ipopt', nlp,
    {'ipopt.print_level':0, 'print_time':False, 'ipopt.tol':1e-10},
    compiler_flags=('-O1',), cache_dir=sys.argv[1], cache_name='split', callbacks=('nlp_g',))
result = solver(x0=[1,1], p=2, lbg=2.5, ubg=ca.inf)
print(json.dumps({'info':info, 'result':{k:v.full().tolist() for k,v in result.items()}}))
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True,
                             text=True, check=True, env=env)
    warm = json.loads(process.stdout.splitlines()[-1])
    assert warm["info"]["hit"] and warm["info"]["callbacks"] == {
        "nlp_g": {**cold["callbacks"]["nlp_g"], "hit": True}
    }
    equal(expected, warm["result"])
    _, changed = build(tmp_path, callbacks=("nlp_g",), compiler_flags=("-O0",))
    assert not changed["hit"]
    altered = problem()
    altered["g"] = altered["g"] + altered["x"][0]
    _, changed_graph = build(tmp_path, altered, callbacks=("nlp_g",))
    assert not changed_graph["hit"]
    manifest = Path(cold["callbacks"]["nlp_g"]["library"]).with_name("manifest.json")
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="Invalid compiled NLP cache entry"):
        build(tmp_path, callbacks=("nlp_g",))


def test_limited_memory_and_custom_derivatives(tmp_path):
    options = {**OPTIONS, "ipopt.hessian_approximation": "limited-memory"}
    native, info = build(tmp_path, options=options)
    assert "nlp_hess_l" not in info["native_callbacks"]
    reference = ca.nlpsol("vm", "ipopt", problem(), options)
    equal(evaluate(reference), evaluate(native))
    # Direct options have precedence over the initial cache inside CasADi.
    options["grad_f"] = reference.get_function("nlp_grad_f")
    options["jac_g"] = reference.get_function("nlp_jac_g")
    native, info = build(tmp_path, options=options)
    assert native.get_function("nlp_grad_f").class_name() == "External"
    assert native.get_function("nlp_jac_g").class_name() == "External"
    equal(evaluate(reference), evaluate(native))


def test_failure_and_options(tmp_path, monkeypatch):
    import bioptim.interfaces.c_compile_cache as cache
    options = Solver.IPOPT()
    options.set_c_compile_callbacks(True, cache_dir=tmp_path, callbacks=["nlp_g"])
    assert options.c_compile and options.c_compile_callbacks and options.compiler_flags == ("-O1",)
    assert options.native_callbacks == ("nlp_g",)
    assert not any("compile" in k or "native_callbacks" in k or "cache" in k
                   for k in options.as_dict(SimpleNamespace(options_common={})))
    options.set_c_compile(True)
    assert not options.c_compile_callbacks and options.native_callbacks is None
    with pytest.raises(ValueError, match="cache_dir"):
        options.set_c_compile_callbacks(True)
    with pytest.raises(ValueError, match="distinct"):
        options.set_c_compile_callbacks(True, cache_dir=tmp_path, callbacks=["nlp_grad"])
    with pytest.raises(ValueError, match="solver-level jit"):
        build(tmp_path, options={**OPTIONS, "jit": True})
    original = cache.subprocess.run
    def fail_compile(command, *args, **kwargs):
        if "-shared" in command:
            return subprocess.CompletedProcess(command, 1, "", "split compile failure")
        return original(command, *args, **kwargs)
    monkeypatch.setattr(cache.subprocess, "run", fail_compile)
    with pytest.raises(RuntimeError, match="split compile failure"):
        build(tmp_path)
    assert not list(tmp_path.iterdir())


def test_generic_solve_switches_callback_configuration(tmp_path, monkeypatch):
    from bioptim.interfaces import interface_utils
    from bioptim.interfaces.solver_interface import SolverInterface
    nlp = problem(False)
    ocp = SimpleNamespace(variables_vector=nlp["x"],
                          bounds_vectors=(np.full((2, 1), -np.inf), np.full((2, 1), np.inf)),
                          init_vector=np.array([[1.0], [1.0]]))
    interface = SolverInterface(ocp)
    interface.opts = Solver.IPOPT()
    interface.opts.set_print_level(0)
    interface.options_common = {"print_time": False}
    interface.solver_name = "IPOPT"
    interface.lam_x = interface.lam_g = None
    interface.shaked_objectives, interface.shaked_constraints = nlp["f"], nlp["g"]
    interface.dispatch_obj_func = lambda: nlp["f"]
    interface.dispatch_bounds = lambda: (nlp["g"], SimpleNamespace(min=2.5, max=np.inf))
    monkeypatch.setattr(interface_utils, "_vectors_are_equal", lambda *args, **kwargs: True)
    reference = interface_utils.generic_solve(interface, False)["sol"]["x"]
    interface.opts.set_c_compile_callbacks(True, cache_dir=tmp_path, callbacks=["nlp_g"])
    actual = interface_utils.generic_solve(interface, False)["sol"]["x"]
    np.testing.assert_allclose(actual, reference, atol=1e-9)
    first = interface.shaked_ocp_solver
    assert first.get_function("nlp_g").class_name() == "External"
    interface_utils.generic_solve(interface, False)
    assert interface.shaked_ocp_solver is first
    interface.opts.set_c_compile_callbacks(True, cache_dir=tmp_path, callbacks=["nlp_f", "nlp_g"])
    interface_utils.generic_solve(interface, False)
    assert interface.shaked_ocp_solver is not first
    assert interface.shaked_ocp_solver.get_function("nlp_f").class_name() == "External"
