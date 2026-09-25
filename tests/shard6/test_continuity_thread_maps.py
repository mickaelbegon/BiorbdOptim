"""Threaded continuity needs one map, while local penalties remain available."""

from pathlib import Path

import casadi as ca
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import ConstraintFcn, OdeSolver, PhaseDynamics
from bioptim.examples.getting_started.basic_ocp import prepare_ocp
from bioptim.interfaces.interface_utils import generic_dispatch_bounds, generic_dispatch_obj_func
from bioptim.interfaces.ipopt_interface import IpoptInterface
from bioptim.limits.penalty_option import PenaltyOption


def _prepare(n_threads, collocation):
    return prepare_ocp(
        biorbd_model_path=str(Path(__file__).parents[2] / "bioptim/examples/models/pendulum.bioMod"),
        final_time=1,
        n_shooting=4,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3) if collocation else OdeSolver.RK4(),
        use_sx=False,
        n_threads=n_threads,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )


def _nlp_probe(ocp):
    interface = IpoptInterface(ocp)
    x = ocp.variables_vector
    g, bounds = generic_dispatch_bounds(interface, include_g=True, include_g_internal=True)
    f = ca.sum1(generic_dispatch_obj_func(interface))
    lam = ca.MX.sym("lam", g.numel())
    sigma = ca.MX.sym("sigma")
    h = ca.hessian(sigma * f + ca.dot(lam, g), x)[0]
    return ca.Function("probe", [x, lam, sigma], [f, g, ca.jacobian(g, x), h]), bounds


@pytest.mark.parametrize("collocation", [False, True])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_continuity_maps_created_once_with_same_nlp(monkeypatch, collocation, n_threads):
    original_set = PenaltyOption._set_penalty_function
    original_map = ca.Function.map
    active_penalty = []
    map_calls = []

    def track_map(function, *args, **kwargs):
        if active_penalty and active_penalty[-1].type == ConstraintFcn.STATE_CONTINUITY:
            map_calls.append(args)
        return original_map(function, *args, **kwargs)

    def track_set(penalty, controllers, fcn):
        active_penalty.append(penalty)
        try:
            return original_set(penalty, controllers, fcn)
        finally:
            active_penalty.pop()

    monkeypatch.setattr(ca.Function, "map", track_map)
    monkeypatch.setattr(PenaltyOption, "_set_penalty_function", track_set)
    optimized = _prepare(n_threads, collocation)
    assert len(map_calls) == (2 if n_threads > 1 else 0)
    optimized_probe, optimized_bounds = _nlp_probe(optimized)
    continuity = next(p for p in optimized.nlp[0].g_internal if p.type == ConstraintFcn.STATE_CONTINUITY)
    assert continuity.multi_thread == (n_threads > 1)
    assert all(continuity.function_non_threaded[node] is not None for node in continuity.node_idx)
    assert all(continuity.weighted_function_non_threaded[node] is not None for node in continuity.node_idx)

    # Recreate the old implementation's unused maps to obtain an independent
    # baseline with the original per-node function layout and dispatch.
    def legacy_set(penalty, controllers, fcn):
        track_set(penalty, controllers, fcn)
        node = controllers[0].node_index
        if (
            n_threads > 1
            and penalty.type == ConstraintFcn.STATE_CONTINUITY
            and penalty.multi_thread
            and len(penalty.node_idx) > 1
            and node != 0
        ):
            penalty.function[node] = original_map(penalty.function[node], len(penalty.node_idx), "thread", n_threads)
            penalty.weighted_function[node] = original_map(
                penalty.weighted_function[node], len(penalty.node_idx), "thread", n_threads
            )
            if penalty.expand:
                penalty.function[node] = penalty.function[node].expand()
                penalty.weighted_function[node] = penalty.weighted_function[node].expand()

    monkeypatch.setattr(PenaltyOption, "_set_penalty_function", legacy_set)
    reference = _prepare(n_threads, collocation)
    reference_probe, reference_bounds = _nlp_probe(reference)
    for i in range(optimized_probe.n_out()):
        assert optimized_probe.sparsity_out(i) == reference_probe.sparsity_out(i)
    npt.assert_array_equal(optimized_bounds.min, reference_bounds.min)
    npt.assert_array_equal(optimized_bounds.max, reference_bounds.max)
    npt.assert_array_equal(optimized.bounds_vectors[0], reference.bounds_vectors[0])
    npt.assert_array_equal(optimized.bounds_vectors[1], reference.bounds_vectors[1])

    rng = np.random.default_rng(42)
    for _ in range(3):
        x = rng.uniform(-0.5, 0.5, optimized_probe.size1_in(0))
        x[0] = 0.25  # positive duration
        lam = rng.normal(size=optimized_probe.size1_in(1))
        sigma = rng.uniform(0.1, 2.0)
        for actual, expected in zip(optimized_probe(x, lam, sigma), reference_probe(x, lam, sigma)):
            npt.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_threaded_continuity_rebuild_refreshes_map(monkeypatch):
    ocp = _prepare(2, collocation=True)
    penalty = next(p for p in ocp.nlp[0].g_internal if p.type == ConstraintFcn.STATE_CONTINUITY)
    previous_map = penalty.weighted_function[0]
    original_map = ca.Function.map
    calls = []

    def track_map(function, *args, **kwargs):
        calls.append(args)
        return original_map(function, *args, **kwargs)

    monkeypatch.setattr(ca.Function, "map", track_map)
    ocp.n_threads = 3
    penalty.add_or_replace_to_penalty_pool(ocp, ocp.nlp[0])
    assert len(calls) == 2
    assert all(args == (4, "thread", 3) for args in calls)
    assert penalty.weighted_function[0] is not previous_map
    assert penalty.multi_thread
