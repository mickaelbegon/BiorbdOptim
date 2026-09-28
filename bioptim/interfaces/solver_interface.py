import numpy as np

from ..misc.parameters_types import Bool, AnyDict, AnyListorDict, CX


class SolverInterface:
    """
    Abstract class for an ocp solver

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
    solver: SolverInterface
        A non-abstract implementation of SolverInterface
    out: dict
        The solution structure

    Methods
    -------
    configure(self, **options)
        Set some options
    solve(self) -> dict
        Solve the prepared ocp
    get_optimized_value(self) -> list[dict] | dict
        Get the previously optimized solution
    start_get_iterations(self)
        Create the necessary folder and create the file to store the iterations while optimizing
    finish_get_iterations(self)
        Close the file where iterations are saved and remove temporary folders
    finalize_objective_value(j: dict) -> MX | SX
        Apply weight and dt to all objective values and convert them to scalar value
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
        """

        self.ocp = ocp
        self.solver = None
        self.out = {}

        # This is to perform long preparation only once (if not changed)
        self.pre_shake_tree_objectives = None
        self.shaked_objectives = None
        self.pre_shake_tree_constraints = None
        self.shaked_constraints = None
        self.shaked_ocp_solver = None

        # Optional, solver-neutral diagnostics of the exact canonical NLP
        # submitted to CasADi.  Keeping this disabled by default is important:
        # ordinary solves must not build or evaluate an additional CasADi
        # function.  Interfaces are long-lived during receding-horizon solves,
        # so the evaluator is cached when the audit is explicitly enabled.
        self.initial_nlp_audit_enabled = False
        self.initial_nlp_audits = []
        self._initial_nlp_constraint_audit_function = None
        self._next_initial_guess_override = None

        # Experimental and off by default.  It only replaces selected scalar
        # functions below a Function.map by exact C externals; the global NLP
        # stays MX and is never compiled as a whole.
        self.compiled_thread_map_external_penalties = frozenset()
        self.compiled_thread_map_external_cache_dir = None
        self._compiled_thread_map_external_functions = {}

    def set_next_initial_guess_override(self, values) -> None:
        """Submit an exact, scaled decision vector to the next solve only."""

        candidate = np.asarray(values, dtype=float).reshape(-1, 1)
        if candidate.size == 0 or not np.all(np.isfinite(candidate)):
            raise ValueError("The one-shot NLP initial guess must be finite and non-empty.")
        self._next_initial_guess_override = candidate.copy()

    def consume_initial_guess_override(self, default_values):
        """Return and clear a one-shot scaled decision-vector override."""

        if self._next_initial_guess_override is None:
            return default_values
        candidate = self._next_initial_guess_override
        self._next_initial_guess_override = None
        expected = np.asarray(default_values).size
        if candidate.size != expected:
            raise ValueError(
                "The one-shot NLP initial guess has "
                f"{candidate.size} values; expected {expected}."
            )
        return candidate

    def enable_initial_nlp_audit(self, enabled: Bool = True) -> None:
        """Enable exact pre-solve evaluations of the submitted ``g(x0)``."""

        self.initial_nlp_audit_enabled = enabled

    def enable_compiled_thread_map_external(self, penalty_names, cache_dir) -> None:
        """Opt in to exact C kernels below selected repeated ``ThreadMap`` penalties.

        This is intentionally an advanced construction-time option. Call it
        before building/solving the NLP.  The default is the unchanged native
        CasADi map path.
        """

        if isinstance(penalty_names, str):
            penalty_names = (penalty_names,)
        names = frozenset(str(name) for name in penalty_names)
        if not names:
            raise ValueError("penalty_names must not be empty")
        if cache_dir is None:
            raise ValueError("cache_dir is required for compiled ThreadMap externals")
        self.compiled_thread_map_external_penalties = names
        self.compiled_thread_map_external_cache_dir = cache_dir
        self._compiled_thread_map_external_functions.clear()

    def build_post_shake_penalty_registry(self, expand: Bool = False, materialize: Bool = True):
        """Describe exact canonical penalty terms without changing the NLP.

        The returned registry is intended for advanced code-generation and
        derivative tooling.  It is invalidated whenever this interface's
        symbolic problem changes, just like a solver cache.
        """

        from .interface_utils import build_post_shake_penalty_registry

        return build_post_shake_penalty_registry(self, expand=expand, materialize=materialize)

    def configure(self, **options):
        """
        Set some options

        Parameters
        ----------
        options: dict
            The dictionary of options
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def show_constraints_jacobian_sparsity(self):
        """
        Show the sparsity of the constraints jacobian
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def solve(self, expand_during_shake_tree: Bool) -> AnyDict:
        """
        Solve the prepared ocp

        Parameters
        ----------
        expand_during_shake_tree: bool
            If the graph should be expanded during the shake tree

        Returns
        -------
        A reference to the solution
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def get_optimized_value(self) -> AnyListorDict:
        """
        Get the previously optimized solution

        Returns
        -------
        A solution or a list of solution depending on the number of phases
        """

        out = []
        for key in self.out.keys():
            out.append(self.out[key])
        return out[0] if len(out) == 1 else out

    def online_optim(self, ocp):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        raise RuntimeError("SolverInterface is an abstract class")

    def transform_penalty_value(self, penalty, nlp, value: CX) -> CX:
        """
        Transform a penalty value before it is dispatched to the solver.

        Solver interfaces may override this hook when their canonical problem
        representation requires an equivalent constraint formulation.
        """
        return value

    def start_get_iterations(self):
        """
        Create the necessary folder and create the file to store the iterations while optimizing
        """

        raise RuntimeError("Get Iteration not implemented for solver")

    def finish_get_iterations(self):
        """
        Close the file where iterations are saved and remove temporary folders
        """

        raise RuntimeError("Get Iteration not implemented for solver")
