"""Optional regressions requiring the biorbd codex/muscle-wrapping backend.

Run with its matching Python and PYTHONPATH; explicitly opt into the legacy
1.9.11 backend using BIOPTIM_ENABLE_EXPERIMENTAL_WRAPPING=1. Stock biorbd skips
these wrapping integration tests; generic surrogate tests remain independent.
"""

import os
from pathlib import Path
import subprocess
import sys

import biorbd_casadi as biorbd
import casadi as ca
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import BiorbdModel, ContactType, ParameterList
from bioptim.examples.toy_examples.muscle_driven_ocp.muscle_wrapping_ocp import (
    CASES,
    GeometryMusclesBiorbdModel,
    errors,
    fit_geometries,
    geometry_values,
    prepare_ocp,
    reference_geometries,
    solve_case,
    validate_reference,
)

pytestmark = pytest.mark.skipif(
    not all(hasattr(biorbd, shape) for shape in ("WrappingCylinder", "WrappingEllipsoid")),
    reason="Requires biorbd codex/muscle-wrapping",
)


@pytest.fixture(scope="module", params=["elbow", "shoulder"])
def fitted_case(request):
    case = CASES[request.param]
    reference = reference_geometries(case)
    fitted, _ = fit_geometries(case, reference)
    return case, reference, fitted


def test_wrapping_reference_derivatives_and_hill_law(fitted_case):
    case, reference, _ = fitted_case
    checks = validate_reference(case, reference)
    assert checks["hill_native_torque_max_abs_error_Nm"] < 1e-10
    assert checks["ad_envelope_max_abs_error_m_per_rad"] < 1e-10
    assert checks["length_gradient_finite_difference_max_abs_error_m_per_rad"] < 2e-6
    assert checks["minimum_wrapped_minus_straight_length_m"] > 1e-3
    # All coordinates are rotational: moment arm units are m/rad.
    assert all(g.nb_q == len(case.start) for g in reference)


def test_wrapping_mlp_independent_values_velocities_and_moment_arms(fitted_case):
    case, reference, fitted = fitted_case
    rng = np.random.default_rng(4399)  # distinct from training and example validation
    q = rng.uniform(case.lower, case.upper, (80, len(case.start))).T
    qdot = rng.uniform(-0.4, 0.4, q.shape)
    discrepancy = errors(geometry_values(fitted, q, qdot), geometry_values(reference, q, qdot))
    assert discrepancy["length_m"]["max_abs"] < 5e-5
    assert discrepancy["velocity_m_per_s"]["max_abs"] < 5e-4
    assert discrepancy["moment_arms_m_per_rad"]["max_abs"] < 5e-4
    assert all(np.isfinite(g.fit_result.normalized_loss) for g in fitted)
    for geometry in fitted:
        q0, qdot0 = q[:, 0], qdot[:, 0]
        torque = np.asarray(geometry.joint_torque(q0, 100)).ravel()
        npt.assert_allclose(torque @ qdot0, -100 * float(geometry.velocity(q0, qdot0)), atol=1e-12)


def test_wrapping_identical_ocp_and_native_replay(fitted_case):
    case, _, fitted = fitted_case
    native_ocp, mlp_ocp = prepare_ocp(case, n_shooting=6), prepare_ocp(case, fitted, n_shooting=6)
    assert "tau" not in native_ocp.nlp[0].controls.keys()
    assert "tau" not in mlp_ocp.nlp[0].controls.keys()
    assert native_ocp.nlp[0].states.shape == mlp_ocp.nlp[0].states.shape == 2 * len(case.start)
    native, native_states = solve_case(case, None, n_shooting=6)
    mlp, mlp_states = solve_case(case, fitted, n_shooting=6)
    for report in (native, mlp):
        assert report["status"] == 0
        assert report["max_abs_nlp_constraints"] < 1e-6
        assert report["native_replay"]["max_abs_q_drift_rad"] < 5e-4
        assert report["activation_min"] >= -1e-10
        assert report["activation_max"] <= 1 + 1e-10
    npt.assert_allclose(mlp_states["q"], native_states["q"], atol=1e-4)
    assert abs(mlp["cost"] - native["cost"]) / native["cost"] < 0.01


def test_wrapping_mlp_torque_hessian_codegen_and_model_reconstruction(fitted_case, tmp_path):
    case, _, fitted = fitted_case
    model = GeometryMusclesBiorbdModel(case.model_path, fitted)
    torque = model.muscle_joint_torque()(model.muscle, model.q, model.qdot, [])
    function = ca.Function(
        "wrapped_mlp",
        [model.q, model.qdot, model.muscle],
        [torque, ca.jacobian(torque, model.q), ca.hessian(ca.sum1(torque), model.q)[0]],
    )
    generator = ca.CodeGenerator("wrapped_mlp.c")
    generator.add(function)
    generator.generate(str(tmp_path) + "/")
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", "wrapped_mlp.c", "-o", "wrapped_mlp.so", "-lm"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    compiled = ca.external("wrapped_mlp", str(tmp_path / "wrapped_mlp.so"))
    q = np.array(case.start) * 0.4
    qdot = np.full(len(case.start), 0.1)
    activation = np.linspace(0.2, 0.4, len(fitted))
    for value, expected in zip(compiled(q, qdot, activation), function(q, qdot, activation)):
        npt.assert_allclose(value, expected, atol=1e-9)
        assert np.isfinite(np.asarray(value)).all()
    constructor, kwargs = model.serialize()
    reconstructed = constructor(**kwargs)
    npt.assert_allclose(
        reconstructed.muscle_joint_torque()(activation, q, qdot, []),
        model.muscle_joint_torque()(activation, q, qdot, []),
        atol=1e-12,
    )


def test_wrapping_model_geometry_validation(fitted_case):
    case, references, _ = fitted_case
    with pytest.raises(ValueError, match="one geometry"):
        GeometryMusclesBiorbdModel(case.model_path, references[:-1])
    with pytest.raises(ValueError, match="one geometry"):
        GeometryMusclesBiorbdModel(case.model_path, references[::-1])
    with pytest.raises(ValueError, match="at least four"):
        prepare_ocp(case, n_shooting=3)


def test_legacy_wrapping_compatibility_is_explicit_and_restricted():
    if biorbd.__version__ != "1.9.11":
        pytest.skip("Only applies to the experimental legacy branch")
    environment = dict(os.environ)
    environment.pop("BIOPTIM_ENABLE_EXPERIMENTAL_WRAPPING", None)
    process = subprocess.run(
        [sys.executable, "-c", "import bioptim"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "biorbd_casadi should be at least version 1.11.1" in process.stderr
    model_path = str(CASES["elbow"].model_path)
    with pytest.raises(NotImplementedError, match="parameters, external forces or contacts"):
        BiorbdModel(model_path, parameters=ParameterList(use_sx=False))
    with pytest.raises(NotImplementedError, match="parameters, external forces or contacts"):
        BiorbdModel(model_path, contact_types=[ContactType.RIGID_IMPLICIT])
    contact_model = CASES["elbow"].model_path.parent / "2segments_2dof_2contacts.bioMod"
    with pytest.raises(NotImplementedError, match="model contacts"):
        BiorbdModel(str(contact_model))
