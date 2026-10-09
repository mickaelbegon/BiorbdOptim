from pathlib import Path
import subprocess

import casadi as ca
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import BiorbdModel, CasadiMLP, MuscleGeometry, MuscleGeometrySpline


def analytic_reference():
    q = ca.MX.sym("q", 3)
    length = 0.4 + 0.03 * q[0] ** 2 + 0.02 * q[0] * q[2] + 0.01 * q[2] ** 3
    return MuscleGeometry(ca.Function("reference", [q], [length]), "analytic", [0, 2])


@pytest.fixture(params=["spline", "mlp"])
def geometry(request):
    reference = analytic_reference()
    if request.param == "spline":
        return MuscleGeometrySpline.fit(reference, [0, 2], [np.linspace(-1, 1, 8)] * 2, np.zeros(3))
    return CasadiMLP(
        3,
        "analytic",
        [0, 2],
        [np.array([[0.2, -0.3], [0.4, 0.1]]), np.array([[0.3, -0.2]])],
        [np.array([0.1, -0.2]), np.array([0.2])],
        [-1, -1],
        [1, 1],
        0.3,
        0.1,
    )


def test_geometry_derivatives_sign_and_power(geometry):
    q = np.array([0.13, 0.42, -0.29])
    qdot = np.array([0.7, 2.1, -0.8])
    step = 1e-5
    basis = np.eye(3)
    finite_jacobian = np.array(
        [(float(geometry.length(q + step * e)) - float(geometry.length(q - step * e))) / (2 * step) for e in basis]
    )
    npt.assert_allclose(np.asarray(geometry.length_jacobian(q)).ravel(), finite_jacobian, atol=1e-10)
    npt.assert_allclose(np.asarray(geometry.moment_arms(q)).ravel(), -finite_jacobian, atol=1e-10)
    finite_velocity = (float(geometry.length(q + step * qdot)) - float(geometry.length(q - step * qdot))) / (2 * step)
    npt.assert_allclose(float(geometry.velocity(q, qdot)), finite_velocity, atol=1e-10)
    force = 150.0
    torque = np.asarray(geometry.joint_torque(q, force)).ravel()
    npt.assert_allclose(torque @ qdot, -force * float(geometry.velocity(q, qdot)), atol=1e-12)
    assert float(geometry.length_jacobian(q)[0, 1]) == 0

    symbol = ca.MX.sym("q", 3)
    hessian = ca.Function("length_hessian", [symbol], [ca.hessian(geometry.length(symbol), symbol)[0]])
    finite_hessian = np.column_stack(
        [
            (
                np.asarray(geometry.length_jacobian(q + step * e)).ravel()
                - np.asarray(geometry.length_jacobian(q - step * e)).ravel()
            )
            / (2 * step)
            for e in basis
        ]
    )
    npt.assert_allclose(hessian(q), finite_hessian, atol=1e-9)
    # Second derivative of a torque already uses THIRD derivatives of length.
    torque_gradient = ca.gradient(geometry.joint_torque(symbol, force)[0], symbol)
    gradient_fun = ca.Function("torque_gradient", [symbol], [torque_gradient])
    torque_hessian = ca.Function("torque_hessian", [symbol], [ca.jacobian(torque_gradient, symbol)])
    finite_third = np.column_stack(
        [
            (np.asarray(gradient_fun(q + step * e)).ravel() - np.asarray(gradient_fun(q - step * e)).ravel())
            / (2 * step)
            for e in basis
        ]
    )
    npt.assert_allclose(torque_hessian(q), finite_third, atol=1e-8)


def test_spline_tensor_order_and_knot_continuity():
    reference = analytic_reference()
    spline = MuscleGeometrySpline.fit(
        reference, [0, 2], [np.linspace(-1, 1, 9), np.linspace(-0.8, 0.8, 7)], np.zeros(3)
    )
    rng = np.random.default_rng(17)
    q = rng.uniform(-0.7, 0.7, (3, 30))
    npt.assert_allclose(spline.length(q), reference.length(q), atol=1e-13)
    npt.assert_allclose(spline.length_jacobian(q), reference.length_jacobian(q), atol=1e-12)
    symbol = ca.MX.sym("q", 3)
    third = ca.Function("third", [symbol], [ca.hessian(spline.joint_torque(symbol, 1)[0], symbol)[0]])
    npt.assert_allclose(third([1e-8, 0, 0.2]), third([-1e-8, 0, 0.2]), atol=1e-8)


def test_mlp_fit_value_and_gradient():
    q = ca.MX.sym("q", 2)
    reference = MuscleGeometry(ca.Function("reference", [q], [0.2 + 0.04 * ca.tanh(0.6 * q[1] - 0.1)]), "test", [1])
    mlp = CasadiMLP.fit(
        reference, [1], [-1], [1], np.zeros(2), n_samples=64, hidden_size=8, max_iterations=300, seed=23
    )
    test_q = np.vstack((np.zeros(21), np.linspace(-0.99, 0.99, 21)))
    npt.assert_allclose(mlp.length(test_q), reference.length(test_q), atol=2e-5)
    npt.assert_allclose(mlp.length_jacobian(test_q), reference.length_jacobian(test_q), atol=1e-4)
    assert np.isfinite(mlp.fit_result.normalized_loss)


def test_six_coordinate_spline():
    q = ca.MX.sym("q", 6)
    length = 0.2 + 0.01 * ca.sumsqr(q) + 0.005 * q[0] * q[5] ** 2
    reference = MuscleGeometry(ca.Function("reference", [q], [length]), "six_dof", range(6))
    spline = MuscleGeometrySpline.fit(reference, range(6), [np.linspace(-0.4, 0.4, 6)] * 6, np.zeros(6))
    points = np.random.default_rng(5).uniform(-0.35, 0.35, (6, 3))
    npt.assert_allclose(spline.length(points), reference.length(points), atol=1e-13)
    npt.assert_allclose(spline.length_jacobian(points), reference.length_jacobian(points), atol=1e-12)


@pytest.mark.parametrize("backend", ["spline", "mlp"])
def test_geometry_in_bioptim_ocp(backend):
    from bioptim import Solver, SolutionMerge
    from bioptim.examples.toy_examples.muscle_driven_ocp.muscle_geometry_surrogates import prepare_geometry_ocp

    path = Path(__file__).resolve().parents[2] / "bioptim/examples/models/arm26.bioMod"
    reference = MuscleGeometry.from_biorbd(BiorbdModel(str(path)), "BRA")
    if backend == "spline":
        geometry = MuscleGeometrySpline.fit(reference, [1], [np.linspace(0.3, 2.2, 15)], [0, 1.25])
    else:
        geometry = CasadiMLP.fit(
            reference, [1], [0.3], [2.2], [0, 1.25], n_samples=128, hidden_size=24, max_iterations=300
        )
    target = float(reference.length([0, 1.2]))
    ocp = prepare_geometry_ocp(path, geometry, target, n_shooting=6)
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solution = ocp.solve(solver)
    assert solution.status == 0
    assert np.max(np.abs(solution.constraints)) < 1e-6
    endpoint = solution.decision_states(to_merge=SolutionMerge.NODES)["q"][:, -1]
    assert abs(float(reference.length(endpoint)) - target) < 1e-6


def test_biorbd_muscle_tendon_reference():
    path = Path(__file__).resolve().parents[2] / "bioptim/examples/models/arm26.bioMod"
    model = BiorbdModel(str(path))
    reference = MuscleGeometry.from_biorbd(model, "BRA")
    assert not model._cached_functions
    index = model.muscle_names.index("BRA")
    for elbow in [0.3, 0.8, 1.3, 2.0]:
        q = np.array([0.2, elbow])
        npt.assert_allclose(reference.length_jacobian(q), model.muscle_length_jacobian()(q, [])[index, :], atol=1e-13)
        step = 1e-5
        finite = (float(reference.length(q + [0, step])) - float(reference.length(q - [0, step]))) / (2 * step)
        npt.assert_allclose(float(reference.length_jacobian(q)[0, 1]), finite, atol=1e-10)
    assert ("muscle_length_jacobian", (), frozenset()) in model._cached_functions
    with pytest.raises(ValueError, match="Unknown muscle"):
        MuscleGeometry.from_biorbd(model, "missing")


def test_geometry_codegen_and_serialization(geometry, tmp_path):
    symbol = ca.MX.sym("q", 3)
    velocity = ca.MX.sym("v", 3)
    function = ca.Function(
        "geometry_probe",
        [symbol, velocity],
        [
            geometry.length(symbol),
            geometry.velocity(symbol, velocity),
            geometry.moment_arms(symbol),
            ca.hessian(geometry.joint_torque(symbol, 100)[0], symbol)[0],
        ],
    )
    generator = ca.CodeGenerator("geometry_probe.c")
    generator.add(function)
    generator.generate(str(tmp_path) + "/")
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", "geometry_probe.c", "-o", "geometry_probe.so", "-lm"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    compiled = ca.external("geometry_probe", str(tmp_path / "geometry_probe.so"))
    expected = function([0.1, 0, 0.3], [0.4, 0, -0.2])
    for result, ref in zip(compiled([0.1, 0, 0.3], [0.4, 0, -0.2]), expected):
        npt.assert_allclose(result, ref, atol=1e-12)
    geometry.length.save(str(tmp_path / "length.casadi"))
    npt.assert_allclose(
        ca.Function.load(str(tmp_path / "length.casadi"))([0.1, 0, 0.3]), geometry.length([0.1, 0, 0.3])
    )


def test_invalid_domains():
    reference = analytic_reference()
    with pytest.raises(ValueError, match="unique"):
        MuscleGeometrySpline.fit(reference, [0, 0], [np.linspace(0, 1, 6)] * 2, np.zeros(3))
    with pytest.raises(ValueError, match="at least"):
        MuscleGeometrySpline.fit(reference, [0], [np.linspace(0, 1, 3)], np.zeros(3))
    with pytest.raises(ValueError, match="strictly increasing"):
        MuscleGeometrySpline.fit(reference, [0], [[0, 1, 2, 3, 4, 4]], np.zeros(3))
    with pytest.raises(ValueError, match="Tensor grid"):
        MuscleGeometrySpline.fit(reference, [0, 2], [np.linspace(0, 1, 6)] * 2, np.zeros(3), max_grid_points=10)
