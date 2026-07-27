from dataclasses import dataclass, field
from typing import Any

import casadi as cas

from .abstract_options import GenericSolver
from ..misc.enums import OnlineOptim, SolverType
from ..misc.parameters_types import (
    AnyDict,
    AnyDictOptional,
    Bool,
    BoolOptional,
    Float,
    Int,
    Str,
)


MADNLP_UNAVAILABLE = (
    "The CasADi MadNLP plugin is not available in the installed CasADi build. "
    "Install a CasADi distribution compiled with MadNLP support or follow the "
    "Bioptim MadNLP installation instructions."
)

MADNLP_LINEAR_SOLVERS = {
    "mumps": "MumpsSolver",
    "mumpssolver": "MumpsSolver",
    "umfpack": "UmfpackSolver",
    "umfpacksolver": "UmfpackSolver",
    "lapack": "LapackCPUSolver",
    "lapack_cpu": "LapackCPUSolver",
    "lapackcpusolver": "LapackCPUSolver",
}


def has_madnlp() -> bool:
    """Return whether the installed CasADi build exposes the MadNLP nlpsol plugin."""
    has_nlpsol = getattr(cas, "has_nlpsol", None)
    if has_nlpsol is not None:
        try:
            return bool(has_nlpsol("madnlp"))
        except RuntimeError:
            return False
    try:
        cas.nlpsol_options("madnlp")
    except RuntimeError:
        return False
    return True


@dataclass
class MADNLP(GenericSolver):
    """Options for CasADi's ``madnlp`` nlpsol plugin.

    Solver-specific options must be nested under CasADi's ``madnlp`` dictionary.
    The plugin embeds a Julia runtime; it is not included in standard CasADi wheels.
    """

    type: SolverType = SolverType.MADNLP
    show_online_optim: BoolOptional = None
    online_optim: OnlineOptim | None = None
    show_options: AnyDictOptional = None
    _tol: Float = 1e-6
    _max_iter: Int = 1000
    _print_level: Int = 3
    _linear_solver: Str = "MumpsSolver"
    _c_compile: Bool = False
    _madnlp_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not has_madnlp():
            raise RuntimeError(MADNLP_UNAVAILABLE)

    @property
    def tol(self) -> Float:
        return self._tol

    @property
    def max_iter(self) -> Int:
        return self._max_iter

    @property
    def print_level(self) -> Int:
        return self._print_level

    @property
    def linear_solver(self) -> Str:
        return self._linear_solver

    @property
    def c_compile(self) -> Bool:
        return self._c_compile

    def set_convergence_tolerance(self, tol: Float) -> None:
        self._tol = tol

    def set_constraint_tolerance(self, tol: Float) -> None:
        # MadNLP has no independent constraint tolerance. Its `tol` is the
        # termination threshold for primal, dual, and complementarity errors.
        self._tol = tol

    def set_maximum_iterations(self, num: Int) -> None:
        self._max_iter = num

    def set_print_level(self, value: Int | Str) -> None:
        levels = {"TRACE": 1, "DEBUG": 2, "INFO": 3, "NOTICE": 4, "WARN": 5, "ERROR": 6}
        if isinstance(value, str):
            try:
                value = levels[value.upper()]
            except KeyError as error:
                raise ValueError("MadNLP print level must be TRACE, DEBUG, INFO, NOTICE, WARN, or ERROR") from error
        if value not in levels.values():
            raise ValueError("MadNLP print level must be an integer from 1 to 6 or a MadNLP log-level name")
        self._print_level = value

    def set_linear_solver(self, value: Str) -> None:
        """Select a CPU linear solver bundled in the current MadNLP ``libMad`` runtime.

        ``libMad`` expects Julia type names rather than the lowercase names shown
        by solver banners. Bioptim accepts both forms and emits the canonical
        value, avoiding the runtime's silent fallback to MUMPS.
        """
        if not isinstance(value, str):
            raise ValueError("MadNLP linear solver must be MUMPS, UMFPACK, or LAPACK CPU")
        key = value.replace("-", "_").replace(" ", "_").lower()
        try:
            self._linear_solver = MADNLP_LINEAR_SOLVERS[key]
        except KeyError as error:
            raise ValueError("MadNLP linear solver must be MUMPS, UMFPACK, or LAPACK CPU") from error

    def set_warm_start_options(self, val: Float = 1e-10) -> None:
        """Enable use of multipliers supplied through CasADi's standard nlpsol inputs."""
        self._madnlp_options["dual_initialized"] = True
        self._madnlp_options["mu_init"] = val

    def set_option_unsafe(self, val: Any, name: Str) -> None:
        """Pass an advanced option to MadNLP; the plugin validates it on construction."""
        self._madnlp_options[name] = val

    def set_c_compile(self, val: Bool) -> None:
        if val:
            raise NotImplementedError(
                "C compilation has not been validated with Bioptim and the current CasADi MadNLP plugin."
            )
        self._c_compile = False

    def as_dict(self, solver) -> AnyDict:
        madnlp = {
            "tol": self._tol,
            "max_iter": self._max_iter,
            "print_level": self._print_level,
            "linear_solver": self._linear_solver,
            **self._madnlp_options,
        }
        return {"madnlp": madnlp, **solver.options_common}
