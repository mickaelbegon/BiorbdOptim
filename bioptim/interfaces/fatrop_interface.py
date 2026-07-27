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
from ..misc.enums import SolverType
from ..misc.parameters_types import Bool, AnyDict, AnyDictOptional
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.solution.solution import Solution


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
