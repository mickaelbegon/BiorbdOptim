import numpy as np
import casadi as cas

from .interface_utils import (
    generic_dispatch_bounds,
    generic_dispatch_obj_func,
    generic_get_all_penalties,
    generic_show_constraints_jacobian_sparsity,
    generic_solve,
)
from .solver_interface import SolverInterface
from ..interfaces import Solver
from ..misc.enums import SolverType
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.solution.solution import Solution


def alpaqa_plugin_available() -> bool:
    """Return whether the installed CasADi build can load the alpaqa nlpsol plugin."""
    if not hasattr(cas, "has_nlpsol"):
        return False
    try:
        return bool(cas.has_nlpsol("alpaqa"))
    except RuntimeError:
        return False


def maximum_bound_violation(value, lower_bound, upper_bound) -> float:
    value = np.asarray(value, dtype=float).reshape(-1)
    lower_bound = np.asarray(lower_bound, dtype=float).reshape(-1)
    upper_bound = np.asarray(upper_bound, dtype=float).reshape(-1)
    if not (value.size == lower_bound.size == upper_bound.size):
        raise ValueError(
            "Alpaqa constraint values and bounds must have identical dimensions."
        )
    if value.size == 0:
        return 0.0
    if not np.all(np.isfinite(value)):
        return float("inf")
    return float(
        np.max(
            np.maximum.reduce(
                (lower_bound - value, value - upper_bound, np.zeros(value.size))
            )
        )
    )


class AlpaqaInterface(SolverInterface):
    """Bioptim interface to CasADi's ``nlpsol(..., 'alpaqa', ...)`` plugin."""

    def __init__(self, ocp):
        super().__init__(ocp)
        self.options_common = {}
        self.opts = Solver.ALPAQA()
        self.solver_name = SolverType.ALPAQA.value
        self.nlp = {}
        self.limits = {}
        self.c_compile = False
        self.lam_g = None
        self.lam_x = None

    def solve(self, expand_during_shake_tree=False):
        if not alpaqa_plugin_available():
            raise RuntimeError(
                "The alpaqa plugin is not available in the installed CasADi build.\n\n"
                "Install a CasADi version compiled with alpaqa support or follow "
                "the Bioptim alpaqa installation instructions."
            )
        out = generic_solve(self, expand_during_shake_tree)
        solution = out["sol"]
        stats = self.shaked_ocp_solver.stats()
        solution["solver_stats"] = stats.copy()
        solution["native_status"] = stats.get("unified_return_status")
        solution["inf_pr"] = maximum_bound_violation(
            solution["g"], self.limits["lbg"], self.limits["ubg"]
        )
        return out

    def online_optim(self, ocp, show_options=None):
        raise NotImplementedError(
            "Alpaqa does not currently support Bioptim online optimization callbacks."
        )

    def show_constraints_jacobian_sparsity(self):
        generic_show_constraints_jacobian_sparsity(self)

    def set_lagrange_multiplier(self, sol: Solution) -> None:
        if sol.lam_g is not None:
            self.lam_g = sol.lam_g

    def dispatch_bounds(self, include_g=True, include_g_internal=True):
        return generic_dispatch_bounds(
            self, include_g=include_g, include_g_internal=include_g_internal
        )

    def dispatch_obj_func(self):
        return generic_dispatch_obj_func(self)

    def get_all_penalties(
        self, nlp: NonLinearProgram, penalties, get_bounds: bool = False
    ):
        return generic_get_all_penalties(
            self, nlp, penalties, scaled=True, get_bounds=get_bounds
        )
