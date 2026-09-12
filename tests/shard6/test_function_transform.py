import platform
import shutil
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from bioptim import Solver
from bioptim.interfaces.function_transform import TRANSFORM_PASSES, transformed_nlpsol


HAS_TRANSFORM = hasattr(ca.Function, "transform")
OPTIONS = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}


def _nlp(cx=ca.SX, parameter=False):
    x = cx.sym("x", 2)
    p = cx.sym("p", 1 if parameter else 0)
    target = p[0] if parameter else 1
    return {
        "x": x,
        "p": p,
        "f": (x[0] - target) ** 2 + (x[1] - 2) ** 2,
        "g": ca.vertcat(ca.sin(x[0]) + x[1], x[0] * x[1]),
    }


def _same_signature(left, right):
    assert left.name_in() == right.name_in()
    assert left.name_out() == right.name_out()
    assert all(left.sparsity_in(i) == right.sparsity_in(i) for i in range(left.n_in()))
    assert all(left.sparsity_out(i) == right.sparsity_out(i) for i in range(left.n_out()))


def test_function_transform_option_and_version_guard(monkeypatch):
    solver = Solver.IPOPT()
    assert not solver.function_transform
    with pytest.raises(TypeError, match="must be a bool"):
        solver.set_function_transform("true")
    if HAS_TRANSFORM:
        solver.set_function_transform(True)
        assert solver.function_transform
        assert "ipopt.function_transform" not in solver.as_dict(SimpleNamespace(options_common={}))
    monkeypatch.setattr(ca, "__version__", "3.7.2")
    with pytest.raises(RuntimeError, match="requires CasADi >= 3.8"):
        solver.set_function_transform(True)
    with pytest.raises(RuntimeError, match="requires CasADi >= 3.8"):
        transformed_nlpsol(_nlp(), OPTIONS)
    solver.set_function_transform(False)
    assert not solver.function_transform


@pytest.mark.skipif(not HAS_TRANSFORM, reason="CasADi 3.8 Function.transform")
@pytest.mark.parametrize("cx", [ca.SX, ca.MX])
@pytest.mark.parametrize("parameter", [False, True])
def test_transformed_callbacks_values_derivatives_and_solve(cx, parameter):
    nlp = _nlp(cx, parameter)
    original = ca.nlpsol("reference", "ipopt", nlp, OPTIONS)
    optimized, info = transformed_nlpsol(nlp, OPTIONS)
    assert info["passes"] == [["simplify", "cse", "ref_count", "const_folding"]]
    assert set(info["callbacks"]) == set(original.get_function())
    rng = np.random.default_rng(38)
    for name in original.get_function():
        left, right = original.get_function(name), optimized.get_function(name)
        _same_signature(left, right)
        expected = left.transform(name, [list(p) for p in TRANSFORM_PASSES], {})
        # Verify the actual IPOPT callbacks, not merely an unused transformation.
        assert right.serialize() == expected.serialize()
        for _ in range(5):
            values = [ca.DM(left.sparsity_in(i), rng.normal(size=left.nnz_in(i))) for i in range(left.n_in())]
            for a, b in zip(left.call(values), right.call(values)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    # AD through transformed primal functions must preserve first/second order derivatives too.
    x, p = nlp["x"], nlp["p"]
    lam = cx.sym("lam", 2)
    derivatives = []
    for solver in (original, optimized):
        f = solver.get_function("nlp_f")(x, p)
        g = solver.get_function("nlp_g")(x, p)
        derivatives.append(
            ca.Function(
                "derivatives",
                [x, p, lam],
                [ca.gradient(f, x), ca.jacobian(g, x), ca.hessian(f + ca.dot(lam, g), x)[0]],
            )
        )
    _same_signature(*derivatives)
    values = [[0.3, 0.9], [1.0] if parameter else [], [0.7, -0.2]]
    for a, b in zip(derivatives[0].call(values), derivatives[1].call(values)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    limits = {"x0": [0.3, 0.9], "p": values[1], "lbg": [-10, -10], "ubg": [10, 10]}
    a, b = original(**limits), optimized(**limits)
    assert original.stats()["success"] and optimized.stats()["success"]
    assert original.stats()["iter_count"] == optimized.stats()["iter_count"]
    for name in ("x", "f", "g", "lam_x", "lam_g"):
        np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not HAS_TRANSFORM, reason="CasADi 3.8 Function.transform")
def test_transformation_supports_limited_memory():
    options = {**OPTIONS, "ipopt.hessian_approximation": "limited-memory"}
    solver, info = transformed_nlpsol(_nlp(), options)
    assert "nlp_hess_l" not in info["callbacks"]
    solver(x0=[0.3, 0.9], lbg=-10, ubg=10)
    assert solver.stats()["success"]


@pytest.mark.skipif(
    not HAS_TRANSFORM or platform.system() != "Linux" or not shutil.which("gcc"),
    reason="CasADi 3.8 and Linux/GCC cache",
)
def test_transformed_native_cache_and_generic_reuse(tmp_path, monkeypatch):
    from bioptim.interfaces import interface_utils
    from bioptim.interfaces.solver_interface import SolverInterface

    nlp = _nlp()
    ocp = SimpleNamespace(
        variables_vector=nlp["x"],
        bounds_vectors=(np.full((2, 1), -10), np.full((2, 1), 10)),
        init_vector=np.array([[0.3], [0.9]]),
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
    interface.dispatch_bounds = lambda: (nlp["g"], SimpleNamespace(min=-10, max=10))
    monkeypatch.setattr(interface_utils, "_vectors_are_equal", lambda *args, **kwargs: True)

    reference = interface_utils.generic_solve(interface, False)["sol"]["x"]
    baseline_solver = interface.shaked_ocp_solver
    interface.opts.set_function_transform(True)
    transformed = interface_utils.generic_solve(interface, False)["sol"]["x"]
    assert interface.shaked_ocp_solver is not baseline_solver
    assert interface.function_transform_info["callbacks"]
    np.testing.assert_allclose(transformed, reference, atol=1e-10)
    vm = interface.shaked_ocp_solver
    interface_utils.generic_solve(interface, False)
    assert interface.shaked_ocp_solver is vm
    interface.opts.set_c_compile(True, compiler_flags=["-O1"], cache_dir=tmp_path)
    native = interface_utils.generic_solve(interface, False)["sol"]["x"]
    assert not interface.c_compile_cache_info["hit"]
    np.testing.assert_allclose(native, reference, atol=1e-10)
    assert interface.shaked_ocp_solver.get_function("nlp_hess_l").class_name() == "External"
    first_key = interface.c_compile_cache_info["key"]
    interface.shaked_ocp_solver = None
    interface_utils.generic_solve(interface, False)
    assert interface.c_compile_cache_info["hit"]
    assert interface.c_compile_cache_info["key"] == first_key
    interface.opts.set_function_transform(False)
    interface_utils.generic_solve(interface, False)
    assert interface.function_transform_info is None
    assert interface.c_compile_cache_info["key"] != first_key
    # Also exercise the historical nonpersistent compilation path.
    monkeypatch.chdir(tmp_path)
    interface.opts.set_function_transform(True)
    interface.opts.set_c_compile(True, compiler_flags=["-O1"])
    native = interface_utils.generic_solve(interface, False)["sol"]["x"]
    np.testing.assert_allclose(native, reference, atol=1e-10)
    assert interface.shaked_ocp_solver.get_function("nlp_hess_l").class_name() == "External"
