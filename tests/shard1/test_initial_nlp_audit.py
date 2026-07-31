import numpy as np
import numpy.testing as npt
import pytest
from casadi import Function

from bioptim import OdeSolver, OrderingStrategy
from bioptim.examples.toy_examples.feature_examples import example_variable_scaling
from bioptim.interfaces.fatrop_interface import FatropInterface
from bioptim.interfaces.ipopt_interface import IpoptInterface
from bioptim.interfaces.madnlp_interface import MadnlpInterface
from bioptim.interfaces import interface_utils
from bioptim.interfaces import madnlp_options
from tests.utils import TestUtils


class _FakeCasadiSolver:
    def __init__(self, nlp):
        self.nlp = nlp
        self.call_limits = []

    def call(self, limits):
        self.call_limits.append(dict(limits))
        return {}

    @staticmethod
    def stats():
        return {"success": True, "iter_count": 0}


@pytest.fixture
def fake_nlpsol(monkeypatch):
    solvers = []

    def factory(_name, _plugin, nlp, _options):
        solver = _FakeCasadiSolver(nlp)
        solvers.append(solver)
        return solver

    monkeypatch.setattr(interface_utils, "nlpsol", factory)
    return solvers


def _prepare_collocation_ocp(use_sx=True):
    return example_variable_scaling.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=3,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
        use_sx=use_sx,
        ordering_strategy=OrderingStrategy.TIME_MAJOR,
    )


def _direct_bound_violation(values, lower_bounds, upper_bounds):
    values = np.asarray(values, dtype=float).reshape(-1, order="F")
    lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(-1, order="F")
    upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1, order="F")
    return np.maximum.reduce((lower_bounds - values, values - upper_bounds, np.zeros(values.size)))


def test_initial_nlp_audit_has_no_evaluator_or_payload_when_disabled(fake_nlpsol):
    interface = IpoptInterface(_prepare_collocation_ocp())

    interface.solve(expand_during_shake_tree=False)

    assert interface.initial_nlp_audits == []
    assert interface._initial_nlp_constraint_audit_function is None
    assert len(fake_nlpsol) == 1


@pytest.mark.parametrize("use_sx", [True, False])
def test_initial_nlp_audit_uses_exact_shaked_collocation_graph_and_caches_function(fake_nlpsol, use_sx):
    interface = IpoptInterface(_prepare_collocation_ocp(use_sx=use_sx))
    interface.enable_initial_nlp_audit()

    interface.solve(expand_during_shake_tree=False)
    first_evaluator = interface._initial_nlp_constraint_audit_function
    first_audit = interface.initial_nlp_audits[-1]
    interface.solve(expand_during_shake_tree=False)
    second_audit = interface.initial_nlp_audits[-1]

    submitted_limits = fake_nlpsol[0].call_limits[0]
    exact_g = Function("test_exact_initial_g", [interface.nlp["x"]], [interface.nlp["g"]])(
        submitted_limits["x0"]
    )
    constraint_violations = _direct_bound_violation(
        exact_g,
        submitted_limits["lbg"],
        submitted_limits["ubg"],
    )
    variable_violations = _direct_bound_violation(
        submitted_limits["x0"],
        submitted_limits["lbx"],
        submitted_limits["ubx"],
    )

    assert first_audit["constraint_function_was_cached"] is False
    assert second_audit["constraint_function_was_cached"] is True
    assert interface._initial_nlp_constraint_audit_function is first_evaluator
    assert first_audit["constraints"]["size"] == int(interface.nlp["g"].shape[0])
    assert first_audit["variables"]["size"] == int(interface.nlp["x"].shape[0])
    assert first_audit["constraints"]["finite"] is True
    assert first_audit["variables"]["finite"] is True
    npt.assert_allclose(
        first_audit["constraints"]["maximum_bound_violation"],
        np.max(constraint_violations),
    )
    npt.assert_allclose(
        first_audit["variables"]["maximum_bound_violation"],
        np.max(variable_violations),
    )
    npt.assert_allclose(
        second_audit["constraints"]["maximum_bound_violation"],
        first_audit["constraints"]["maximum_bound_violation"],
    )


def test_madnlp_initial_nlp_audit_matches_ipopt_canonical_collocation_nlp(fake_nlpsol, monkeypatch):
    ocp = _prepare_collocation_ocp()
    ipopt = IpoptInterface(ocp)
    ipopt.enable_initial_nlp_audit()
    ipopt.solve(expand_during_shake_tree=False)

    # Constructing MadnlpInterface normally certifies the external plugin.
    # The fake nlpsol isolates this unit test from that optional runtime while
    # still exercising MadNLP's real dispatch/get_all_penalties methods.
    monkeypatch.setattr(madnlp_options, "has_madnlp", lambda: True)
    madnlp = MadnlpInterface(ocp)
    madnlp.enable_initial_nlp_audit()
    madnlp.solve(expand_during_shake_tree=False)

    ipopt_audit = ipopt.initial_nlp_audits[-1]
    madnlp_audit = madnlp.initial_nlp_audits[-1]
    for section in ("constraints", "variables"):
        assert madnlp_audit[section]["size"] == ipopt_audit[section]["size"]
        npt.assert_allclose(
            madnlp_audit[section]["maximum_bound_violation"],
            ipopt_audit[section]["maximum_bound_violation"],
        )
        npt.assert_allclose(
            madnlp_audit[section]["l2_bound_violation"],
            ipopt_audit[section]["l2_bound_violation"],
        )


def test_initial_nlp_audit_uses_solver_call_limits_without_mutating_interface_limits(fake_nlpsol):
    interface = FatropInterface(_prepare_collocation_ocp())
    interface.opts.set_bound_tightening_factor(1e-8)
    interface.ocp.nlp[0].u_init["tau"].init[0, :] = 1000.0
    interface.enable_initial_nlp_audit()

    interface.solve(expand_during_shake_tree=False)

    raw_x0 = np.asarray(interface.limits["x0"], dtype=float).copy()
    submitted_x0 = np.asarray(fake_nlpsol[0].call_limits[0]["x0"], dtype=float)
    npt.assert_allclose(np.asarray(interface.limits["x0"], dtype=float), raw_x0)
    assert np.max(np.abs(raw_x0 - submitted_x0)) > 0.0
    assert interface.initial_nlp_audits[-1]["variables"]["maximum_bound_violation"] == 0.0
