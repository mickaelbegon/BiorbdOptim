"""Differentiable, fixed-parameter muscle-tendon geometry surrogates.

Lengths are in metres and velocities are positive during lengthening. The moment
arm matrix is -dl/dq (muscles by coordinates), hence tau = moment_arms.T @ force.
For rotational coordinates moment arms have units m/rad. These objects do not
replace a Hill muscle force law or BiorbdModel.muscle_joint_torque().
"""

from dataclasses import dataclass
from math import prod

import casadi as ca
import numpy as np


class MuscleGeometry:
    """CasADi functions derived from a single scalar muscle-tendon length.

    The coordinates must have the same derivative as the generalized velocities
    (no quaternions). Use the fitting domain as OCP bounds: evaluation outside it
    is not validated, and no clipping is performed inside the symbolic graph.
    Fitting fixes unselected coordinates at q_reference; only select coordinates
    that capture the muscle's geometric dependencies.
    """

    def __init__(self, length: ca.Function, muscle_name: str, dof_indices, lower=None, upper=None):
        self.muscle_name = muscle_name
        self.nb_q = length.size1_in(0)
        if length.n_in() != 1 or length.n_out() != 1 or length.size_out(0) != (1, 1):
            raise ValueError("Expected a length function with one vector input and one scalar output")
        indices = tuple(dof_indices)
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= self.nb_q for i in indices):
            raise ValueError("dof_indices must be unique valid coordinate indices")
        self.dof_indices = indices
        self.lower = None if lower is None else np.asarray(lower, dtype=float).copy()
        self.upper = None if upper is None else np.asarray(upper, dtype=float).copy()
        self.length = length
        q = ca.MX.sym("q", self.nb_q)
        qdot = ca.MX.sym("qdot", self.nb_q)
        force = ca.MX.sym("force")
        jacobian = ca.jacobian(length(q), q)
        self.length_jacobian = ca.Function("length_jacobian", [q], [jacobian], ["q"], ["dl_dq"])
        self.moment_arms = ca.Function("moment_arms", [q], [-jacobian], ["q"], ["r"])
        self.velocity = ca.Function("lengthening_velocity", [q, qdot], [jacobian @ qdot])
        self.joint_torque = ca.Function("muscle_joint_torque", [q, force], [-jacobian.T * force])

    @classmethod
    def from_biorbd(cls, model, muscle_name: str):
        """Build a reference using a fresh model, preserving the caller's caches.

        This is muscle-TENDON length, not fibre length (which depends on tendon
        compliance/pennation). Generalized coordinates and velocities must agree.
        Parametrized models are deliberately rejected: weights are frozen.
        """
        import biorbd_casadi as biorbd

        if model.nb_quaternions or model.nb_q != model.nb_qdot:
            raise NotImplementedError("Geometry requires qdot = dq/dt; quaternion models are not supported")
        if model.parameters.numel():
            raise NotImplementedError("Fit fixed geometry parameters; parametrized models are not supported")
        raw_model = biorbd.Model(model.path)
        names = tuple(name.to_string() for name in raw_model.muscleNames())
        if muscle_name not in names:
            raise ValueError(f"Unknown muscle {muscle_name!r}; available muscles: {names}")
        q = ca.MX.sym("q", model.nb_q)
        coordinates = biorbd.GeneralizedCoordinates(q)
        # Updating all muscles first is essential: calling muscle.length with an
        # un-updated joint model can silently produce a constant CasADi graph.
        raw_model.updateMuscles(coordinates, True)
        length = raw_model.muscle(names.index(muscle_name)).musculoTendonLength(raw_model, coordinates, False).to_mx()
        function = ca.Function("reference_muscle_tendon_length", [q], [length], ["q"], ["l_mt"]).expand()
        return cls(function, muscle_name, range(model.nb_q))


def _validate_domain(reference, dof_indices, lower, upper, q_reference):
    indices = tuple(dof_indices)
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= reference.nb_q for i in indices):
        raise ValueError("dof_indices must be unique valid coordinate indices")
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    q_reference = np.asarray(q_reference, dtype=float)
    if lower.shape != (len(indices),) or upper.shape != lower.shape:
        raise ValueError("Provide one lower and upper bound per active coordinate")
    if q_reference.shape != (reference.nb_q,):
        raise ValueError("q_reference must contain all model coordinates")
    if not all(np.isfinite(v).all() for v in (lower, upper, q_reference)) or np.any(lower >= upper):
        raise ValueError("The domain must be finite with lower < upper")
    return indices, lower, upper, q_reference


def _full_coordinates(reference, indices, reduced, q_reference):
    q = np.repeat(q_reference[:, None], reduced.shape[1], axis=1)
    q[list(indices)] = reduced
    return q


class MuscleGeometrySpline(MuscleGeometry):
    """Tensor-product interpolating B-spline, quintic by default.

    A degree-5 spline with simple interior knots is C4. This matters when exact
    OCP Hessians differentiate the moment arms twice (third derivatives of the
    length). Cubic splines are only C2. Tensor grids cost n_grid**n_active; exploit
    anatomical sparsity and use a MLP for large numbers of active coordinates.
    """

    @classmethod
    def fit(cls, reference, dof_indices, grids, q_reference, degree=5, max_grid_points=1_000_000):
        from scipy.interpolate import make_interp_spline

        grids = [np.asarray(grid, dtype=float) for grid in grids]
        if not grids or any(grid.ndim != 1 or len(grid) < degree + 1 for grid in grids):
            raise ValueError("Each grid must be a vector containing at least degree + 1 points")
        if degree not in (3, 5):
            raise ValueError("Supported spline degrees are 3 and 5")
        if any(not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0) for grid in grids):
            raise ValueError("Spline grids must be finite and strictly increasing")
        indices, lower, upper, q_reference = _validate_domain(
            reference, dof_indices, [g[0] for g in grids], [g[-1] for g in grids], q_reference
        )
        count = prod(len(g) for g in grids)
        if count > max_grid_points:
            raise ValueError(f"Tensor grid contains {count} points; reduce dimensions or use CasadiMLP")
        mesh = np.meshgrid(*grids, indexing="ij")
        reduced = np.vstack([axis.ravel(order="F") for axis in mesh])
        full = _full_coordinates(reference, indices, reduced, q_reference)
        values = np.empty(count)
        for start in range(0, count, 1024):
            values[start : start + 1024] = np.asarray(reference.length(full[:, start : start + 1024])).ravel()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite reference lengths on the spline grid")
        coefficients = values.reshape(tuple(len(g) for g in grids), order="F")
        knots = []
        # Separable interpolation avoids a huge multidimensional sparse solve.
        for axis, grid in enumerate(grids):
            spline = make_interp_spline(grid, coefficients, k=degree, axis=axis)
            coefficients = np.moveaxis(spline.c, 0, axis)
            knots.append(spline.t.tolist())
        q = ca.MX.sym("q", reference.nb_q)
        expression = ca.bspline(
            q[list(indices)], coefficients.ravel(order="F").tolist(), knots, [degree] * len(grids), 1, {}
        )
        length = ca.Function("spline_muscle_tendon_length", [q], [expression], ["q"], ["l_mt"])
        fitted = cls(length, reference.muscle_name, indices, lower, upper)
        fitted.grid_point_count = count
        fitted.degree = degree
        fitted.q_reference = q_reference.copy()
        return fitted


@dataclass
class MLPFitResult:
    iterations: int
    success: bool
    message: str
    normalized_loss: float


class CasadiMLP(MuscleGeometry):
    """One-hidden-layer tanh MLP with normalisation in the CasADi graph.

    Fit in float64 using SciPy L-BFGS, supervising BOTH lengths and dl/dq. There
    is no Torch dependency at inference or fitting, no dropout, and no mutable
    hidden state. All weights, biases and normalisations are frozen at inference.
    """

    def __init__(self, nb_q, muscle_name, dof_indices, weights, biases, lower, upper, output_mean, output_scale):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if lower.shape != (len(dof_indices),) or upper.shape != lower.shape or np.any(lower >= upper):
            raise ValueError("Provide increasing bounds for each active coordinate")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Bounds must be finite")
        if not np.isfinite(output_scale) or output_scale <= 0 or not np.isfinite(output_mean):
            raise ValueError("Output normalisation must be finite with positive scale")
        weights = [np.asarray(w, dtype=float).copy() for w in weights]
        biases = [np.asarray(b, dtype=float).reshape(-1, 1).copy() for b in biases]
        if len(weights) != 2 or len(biases) != 2:
            raise ValueError("Expected two affine layers")
        hidden = weights[0].shape[0]
        if (
            weights[0].shape != (hidden, len(dof_indices))
            or weights[1].shape != (1, hidden)
            or biases[0].shape != (hidden, 1)
            or biases[1].shape != (1, 1)
            or not all(np.isfinite(v).all() for v in weights + biases)
        ):
            raise ValueError("Inconsistent or non-finite MLP weights and biases")
        q = ca.MX.sym("q", nb_q)
        x = (q[list(dof_indices)] - ca.DM((upper + lower) / 2)) / ca.DM((upper - lower) / 2)
        hidden_expression = ca.tanh(ca.DM(weights[0]) @ x + ca.DM(biases[0]))
        expression = output_mean + output_scale * (ca.DM(weights[1]) @ hidden_expression + ca.DM(biases[1]))
        function = ca.Function("mlp_muscle_tendon_length", [q], [expression], ["q"], ["l_mt"])
        super().__init__(function, muscle_name, dof_indices, lower, upper)
        self.weights = weights
        self.biases = biases
        self.output_mean = output_mean
        self.output_scale = output_scale
        self.fit_result = None

    @classmethod
    def fit(
        cls,
        reference,
        dof_indices,
        lower,
        upper,
        q_reference,
        n_samples=1024,
        hidden_size=64,
        max_iterations=1000,
        jacobian_weight=1.0,
        seed=42,
    ):
        from scipy.linalg import lstsq
        from scipy.optimize import minimize
        from scipy.stats import qmc

        indices, lower, upper, q_reference = _validate_domain(reference, dof_indices, lower, upper, q_reference)
        if n_samples < 2 or hidden_size < 1 or max_iterations < 1 or jacobian_weight < 0:
            raise ValueError("Invalid sample count, hidden size, iterations or Jacobian weight")
        dimension = len(indices)
        normalized_x = 2 * qmc.LatinHypercube(dimension, seed=seed).random(n_samples) - 1
        input_scale = (upper - lower) / 2
        reduced = ((upper + lower) / 2 + normalized_x * input_scale).T
        full = _full_coordinates(reference, indices, reduced, q_reference)
        lengths = np.asarray(reference.length(full)).ravel()
        q = ca.MX.sym("q", reference.nb_q)
        reduced_jacobian = ca.Function("reduced_jacobian", [q], [reference.length_jacobian(q)[:, list(indices)]])
        derivatives = np.asarray(reduced_jacobian(full)).reshape(n_samples, dimension)
        if not np.isfinite(lengths).all() or not np.isfinite(derivatives).all():
            raise ValueError("Non-finite reference training data")
        output_mean = float(np.mean(lengths))
        output_scale = max(float(np.std(lengths)), 1e-6)
        target = (lengths - output_mean) / output_scale
        target_jacobian = derivatives * input_scale / output_scale
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.5 / np.sqrt(dimension), (hidden_size, dimension))
        b = rng.normal(0, 0.5, hidden_size)
        a = np.tanh(normalized_x @ w.T + b)
        # Warm start the output layer by solving a value+derivative least-squares fit.
        features = [np.column_stack((a, np.ones(n_samples)))]
        targets = [target]
        derivative_scale = np.sqrt(jacobian_weight / dimension)
        for axis in range(dimension):
            features.append(np.column_stack(((1 - a**2) * w[:, axis], np.zeros(n_samples))) * derivative_scale)
            targets.append(target_jacobian[:, axis] * derivative_scale)
        output = lstsq(np.vstack(features), np.concatenate(targets), cond=1e-7)[0]
        initial = np.concatenate((w.ravel(), b, output))
        regularization = 1e-10

        def unpack(parameters):
            w_end = hidden_size * dimension
            return (
                parameters[:w_end].reshape(hidden_size, dimension),
                parameters[w_end : w_end + hidden_size],
                parameters[w_end + hidden_size : -1],
                parameters[-1],
            )

        def loss_and_gradient(parameters):
            w, b, v, c = unpack(parameters)
            a = np.tanh(normalized_x @ w.T + b)
            slope = 1 - a**2
            value_error = a @ v + c - target
            derivative_error = (slope * v) @ w - target_jacobian
            loss = np.mean(value_error**2) + jacobian_weight * np.mean(derivative_error**2)
            value_adjoint = 2 * value_error / n_samples
            derivative_adjoint = 2 * jacobian_weight * derivative_error / (n_samples * dimension)
            derivative_projection = derivative_adjoint @ w.T
            hidden_adjoint = value_adjoint[:, None] * slope * v
            hidden_adjoint += -2 * a * slope * v * derivative_projection
            grad_w = hidden_adjoint.T @ normalized_x + (slope * v).T @ derivative_adjoint
            grad_b = hidden_adjoint.sum(axis=0)
            grad_v = a.T @ value_adjoint + np.sum(slope * derivative_projection, axis=0)
            gradient = np.concatenate((grad_w.ravel(), grad_b, grad_v, [value_adjoint.sum()]))
            return loss + regularization * np.sum(parameters**2), gradient + 2 * regularization * parameters

        result = minimize(
            loss_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-8, "maxls": 40},
        )
        w, b, v, c = unpack(result.x)
        fitted = cls(
            reference.nb_q,
            reference.muscle_name,
            indices,
            [w, v[None, :]],
            [b, [c]],
            lower,
            upper,
            output_mean,
            output_scale,
        )
        fitted.fit_result = MLPFitResult(result.nit, bool(result.success), str(result.message), float(result.fun))
        fitted.q_reference = q_reference.copy()
        return fitted
