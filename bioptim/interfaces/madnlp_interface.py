from .interface_utils import (
    generic_dispatch_bounds,
    generic_dispatch_obj_func,
    generic_get_all_penalties,
    generic_set_lagrange_multiplier,
    generic_solve,
)
from .solver_interface import SolverInterface
from ..interfaces import Solver
from ..misc.enums import SolverType
from ..misc.parameters_types import AnyDict, AnyDictOptional, Bool
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.solution.solution import Solution


class MadnlpInterface(SolverInterface):
    """Bioptim interface to CasADi's MadNLP nlpsol plugin."""

    def __init__(self, ocp):
        super().__init__(ocp)
        self.options_common = {}
        self.opts = Solver.MADNLP()
        self.solver_name = SolverType.MADNLP.value
        self.nlp = {}
        self.limits = {}
        self.ocp_solver = None
        self.c_compile = False
        self.lam_g = None
        self.lam_x = None

    def online_optim(self, ocp, show_options: AnyDictOptional = None):
        raise NotImplementedError("MadNLP does not currently support Bioptim online optimization callbacks.")

    def solve(self, expand_during_shake_tree: Bool) -> AnyDict:
        return generic_solve(self, expand_during_shake_tree)

    def set_lagrange_multiplier(self, sol: Solution) -> None:
        generic_set_lagrange_multiplier(self, sol)

    def dispatch_bounds(self, include_g: Bool = True, include_g_internal: Bool = True):
        return generic_dispatch_bounds(self, include_g=include_g, include_g_internal=include_g_internal)

    def dispatch_obj_func(self):
        return generic_dispatch_obj_func(self)

    def get_all_penalties(self, nlp: NonLinearProgram, penalties, get_bounds: bool = False):
        return generic_get_all_penalties(self, nlp, penalties, scaled=True, get_bounds=get_bounds)
