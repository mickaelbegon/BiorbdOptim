from casadi import DM
import numpy as np

from .interface_utils import (
    generic_show_constraints_jacobian_sparsity,
    generic_solve,
    generic_dispatch_bounds,
    generic_dispatch_obj_func,
    generic_get_all_penalties,
    generic_set_lagrange_multiplier,
)
from .solver_interface import SolverInterface
from ..interfaces import Solver
from ..limits.constraints import ConstraintFcn
from ..limits.phase_transition import PhaseTransitionFcn
from ..misc.enums import DefectType, SolverType
from ..misc.parameters_types import Bool, AnyDict, AnyDictOptional, CX
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.solution.solution import Solution
from ..optimization.vector_layout import OrderingStrategy


class FatropInterface(SolverInterface):
    """
    The Fatrop solver interface

    Attributes
    ----------
    options_common: dict
        Options irrelevant of a specific ocp
    opts: FATROP
        Options of the current ocp
    nlp: dict
        The declaration of the variables Fatrop-friendly
    limits: dict
        The declaration of the bound Fatrop-friendly
    lam_g: np.ndarray
        The lagrange multiplier of the constraints to initialize the solver
    lam_x: np.ndarray
        The lagrange multiplier of the variables to initialize the solver

    Methods
    -------
    online_optim(self, ocp: OptimalControlProgram)
        Declare the online callback to update the graphs while optimizing
    solve(self) -> dict
        Solve the prepared ocp
    set_lagrange_multiplier(self, sol: dict)
        Set the lagrange multiplier from a solution structure
    __dispatch_bounds(self)
        Parse the bounds of the full ocp to a Ipopt-friendly one
    __dispatch_obj_func(self)
        Parse the objective functions of the full ocp to a Ipopt-friendly one
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        super().__init__(ocp)

        if ocp.vector_layout.ordering == OrderingStrategy.VARIABLE_MAJOR:
            raise ValueError(
                "FATROP requires OrderingStrategy.TIME_MAJOR so decision variables follow "
                "[x_0, u_0, x_1, u_1, ...]."
            )

        self.options_common = {}
        self.opts = Solver.FATROP()
        self.solver_name = SolverType.FATROP.value

        self.nlp = {}
        self.limits = {}
        self.ocp_solver = None
        self.c_compile = False

        self.lam_g = None
        self.lam_x = None

    def online_optim(self, ocp, show_options: AnyDictOptional = None):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        show_options: dict
            The options to pass to PlotOcp
        """

        raise NotImplementedError("Fatrop does not support online optimization yet.")

    def show_constraints_jacobian_sparsity(self):
        """
        Show the sparsity of the constraints jacobian
        """
        generic_show_constraints_jacobian_sparsity(self)

    def solve(self, expand_during_shake_tree: Bool) -> AnyDict:
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """
        return generic_solve(self, expand_during_shake_tree)

    def solver_call_limits(self) -> dict:
        """Tighten interval bounds to compensate Fatrop's relative relaxation.

        The original ``self.limits`` remain untouched so callers can audit the
        returned point against the physical OCP bounds.
        """

        factor = self.opts.bound_tightening_factor
        if factor == 0:
            return self.limits

        lower = np.asarray(self.limits["lbx"], dtype=float).reshape(-1)
        upper = np.asarray(self.limits["ubx"], dtype=float).reshape(-1)
        initial = np.asarray(self.limits["x0"], dtype=float).reshape(-1)
        interval = lower < upper

        tightened_lower = lower.copy()
        tightened_upper = upper.copy()
        finite_lower = interval & np.isfinite(lower)
        finite_upper = interval & np.isfinite(upper)
        tightened_lower[finite_lower] += factor * np.maximum(
            1.0, np.abs(lower[finite_lower])
        )
        tightened_upper[finite_upper] -= factor * np.maximum(
            1.0, np.abs(upper[finite_upper])
        )
        if np.any(tightened_lower > tightened_upper):
            raise ValueError(
                "Fatrop bound tightening exceeds at least one decision interval."
            )

        call_limits = dict(self.limits)
        call_limits["lbx"] = DM(tightened_lower)
        call_limits["ubx"] = DM(tightened_upper)
        call_limits["x0"] = DM(
            np.minimum(np.maximum(initial, tightened_lower), tightened_upper)
        )
        return call_limits

    def set_lagrange_multiplier(self, sol: Solution) -> None:
        """
        Set the lagrange multiplier from a solution structure

        Parameters
        ----------
        sol: dict
            A solution structure where the lagrange multipliers are set
        """
        sol = generic_set_lagrange_multiplier(self, sol)

    def dispatch_bounds(self, include_g: Bool = True, include_g_internal: Bool = True):
        """
        Parse the bounds of the full ocp to a Ipopt-friendly one
        """
        return generic_dispatch_bounds(self, include_g=include_g, include_g_internal=include_g_internal)

    def dispatch_obj_func(self):
        """
        Parse the objective functions of the full ocp to a Ipopt-friendly one

        Returns
        -------
        SX | MX
            The objective function
        """
        return generic_dispatch_obj_func(self)

    def get_all_penalties(self, nlp: NonLinearProgram, penalties, get_bounds: bool = False):
        """
        Parse the penalties of the full ocp to a Ipopt-friendly one

        Parameters
        ----------
        nlp: NonLinearProgram
            The nonlinear program to parse the penalties from
        penalties:
            The penalties to parse
        get_bounds: bool
            If the bounds should also be returned. This can only be used if the penalties are constraints

        Returns
        -------

        """
        return generic_get_all_penalties(self, nlp, penalties, scaled=True, get_bounds=get_bounds)

    @staticmethod
    def _state_scaling(nlp: NonLinearProgram) -> np.ndarray:
        return np.concatenate([nlp.x_scaling[key].scaling for key in nlp.states.keys()])

    @staticmethod
    def _repeat_penalty_scaling(penalty, value: CX, scaling_per_node: np.ndarray) -> np.ndarray:
        n_nodes = len(penalty.node_idx) if penalty.multi_thread else 1
        if value.shape[0] != scaling_per_node.shape[0] * n_nodes:
            raise RuntimeError(
                f"Cannot normalize FATROP constraint {penalty.name}: expected "
                f"{scaling_per_node.shape[0] * n_nodes} rows, got {value.shape[0]}."
            )
        return np.tile(scaling_per_node, (n_nodes, 1))

    def _state_continuity_scaling(self, penalty, nlp: NonLinearProgram, value: CX) -> np.ndarray:
        state_scaling = self._state_scaling(nlp)
        n_states = state_scaling.shape[0]
        n_nodes = len(penalty.node_idx) if penalty.multi_thread else 1
        if value.shape[0] % n_nodes:
            raise RuntimeError(
                f"Cannot normalize FATROP constraint {penalty.name}: "
                f"{value.shape[0]} rows cannot be split over {n_nodes} nodes."
            )

        rows_per_node = value.shape[0] // n_nodes
        if rows_per_node < n_states:
            raise RuntimeError(
                f"Cannot normalize FATROP constraint {penalty.name}: "
                f"{rows_per_node} rows are fewer than the {n_states} states."
            )

        scaling_per_node = np.ones((rows_per_node, 1))
        scaling_per_node[:n_states] = state_scaling

        ode_solver = nlp.dynamics_type.ode_solver
        if ode_solver.is_direct_collocation and ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            n_collocation_points = ode_solver.polynomial_degree
            n_defect_rows = rows_per_node - n_states
            if n_defect_rows % n_collocation_points:
                raise RuntimeError(
                    f"Cannot normalize FATROP collocation defects for {penalty.name}: "
                    f"{n_defect_rows} rows cannot be split over {n_collocation_points} collocation points."
                )

            rows_per_defect = n_defect_rows // n_collocation_points
            if rows_per_defect < n_states:
                raise RuntimeError(
                    f"Cannot normalize FATROP collocation defects for {penalty.name}: "
                    f"each defect has {rows_per_defect} rows for {n_states} states."
                )
            for collocation_point in range(n_collocation_points):
                first_row = n_states + collocation_point * rows_per_defect
                scaling_per_node[first_row : first_row + n_states] = state_scaling

        return self._repeat_penalty_scaling(penalty, value, scaling_per_node)

    def transform_penalty_value(self, penalty, nlp, value: CX) -> CX:
        """
        Express FATROP's state gap-closing constraints in scaled coordinates.

        FATROP requires the Jacobian block of every gap-closing constraint with
        respect to the next state to be exactly the identity. Bioptim's decision
        variables are scaled, while its dynamics functions remain in physical
        coordinates, so the unmodified block is the state scaling matrix.
        Consequently, FATROP's constraint tolerance applies in scaled state
        coordinates for these rows.
        """
        if not isinstance(nlp, NonLinearProgram):
            return value

        if penalty.type == ConstraintFcn.STATE_CONTINUITY:
            return value / self._state_continuity_scaling(penalty, nlp, value)

        if penalty.type == ConstraintFcn.FIRST_COLLOCATION_HELPER_EQUALS_STATE:
            state_scaling = self._state_scaling(nlp)
            return value / self._repeat_penalty_scaling(penalty, value, state_scaling)

        if penalty.type in (PhaseTransitionFcn.CONTINUOUS, PhaseTransitionFcn.IMPACT):
            phase_pre, phase_post = (phase_idx % self.ocp.n_phases for phase_idx in penalty.nodes_phase)
            if phase_post != phase_pre + 1:
                raise RuntimeError(
                    f"FATROP cannot represent the non-sequential phase transition {phase_pre}->{phase_post} "
                    "as a gap-closing constraint."
                )
            post_state_scaling = self._state_scaling(self.ocp.nlp[phase_post])
            return -value / self._repeat_penalty_scaling(penalty, value, post_state_scaling)

        return value
