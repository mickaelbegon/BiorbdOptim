from dataclasses import dataclass, field
from typing import Any

from ..misc.enums import OnlineOptim, SolverType
from ..misc.parameters_types import AnyDictOptional, BoolOptional, Float, Int, Str
from .abstract_options import GenericSolver

_SUPPORTED_OPTIONS = {
    "alm": {
        "tolerance",
        "dual_tolerance",
        "initial_penalty",
        "initial_penalty_factor",
        "initial_tolerance",
        "tolerance_update_factor",
        "penalty_update_factor",
        "rel_penalty_increase_threshold",
        "max_multiplier",
        "max_penalty",
        "min_penalty",
        "max_iter",
        "max_time",
        "print_interval",
        "print_precision",
        "single_penalty_factor",
    },
    "panoc": {
        "max_iter",
        "max_time",
        "min_linesearch_coefficient",
        "force_linesearch",
        "linesearch_strictness_factor",
        "L_min",
        "L_max",
        "stop_crit",
        "max_no_progress",
        "print_interval",
        "print_precision",
        "quadratic_upperbound_tolerance_factor",
        "linesearch_tolerance_factor",
        "disable_acceleration",
    },
    "lbfgs": {
        "memory",
    },
}


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Alpaqa option '{name}' must be a number.")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Alpaqa option '{name}' must be {qualifier}.")


@dataclass
class ALPAQA(GenericSolver):
    """Options for CasADi's alpaqa nlpsol plugin (ALM with PANOC and L-BFGS)."""

    type: SolverType = SolverType.ALPAQA
    show_online_optim: BoolOptional = None
    online_optim: OnlineOptim | None = None
    show_options: AnyDictOptional = None
    _c_compile: bool = False
    _alpaqa_options: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"alm": {}, "panoc": {}, "lbfgs": {}}
    )

    @property
    def c_compile(self) -> bool:
        return self._c_compile

    def _set_option(self, group: str, name: str, value: Any) -> None:
        if group not in _SUPPORTED_OPTIONS or name not in _SUPPORTED_OPTIONS[group]:
            raise ValueError(
                f"Unknown alpaqa option '{group}.{name}'. "
                'Use casadi.nlpsol_options("alpaqa") and the alpaqa parameter documentation to inspect supported options.'
            )
        self._alpaqa_options[group][name] = value

    def set_option_unsafe(self, val: Any, name: Str) -> None:
        try:
            group, option = name.split(".", 1)
        except ValueError as exc:
            raise ValueError(
                "Alpaqa options must use the '<alm|panoc|lbfgs>.<name>' form."
            ) from exc
        self._set_option(group, option, val)

    def set_alm_option(self, name: Str, value: Any) -> None:
        self._set_option("alm", name, value)

    def set_panoc_option(self, name: Str, value: Any) -> None:
        self._set_option("panoc", name, value)

    def set_lbfgs_option(self, name: Str, value: Any) -> None:
        self._set_option("lbfgs", name, value)

    def set_convergence_tolerance(self, tol: Float) -> None:
        _positive_number(tol, "alm.tolerance")
        self.set_alm_option("tolerance", tol)

    def set_constraint_tolerance(self, tol: Float) -> None:
        _positive_number(tol, "alm.dual_tolerance")
        self.set_alm_option("dual_tolerance", tol)

    def set_maximum_iterations(self, num: Int) -> None:
        _positive_number(num, "panoc.max_iter")
        if not isinstance(num, int):
            raise TypeError("Alpaqa option 'panoc.max_iter' must be an integer.")
        self.set_panoc_option("max_iter", num)

    def set_alm_maximum_iterations(self, num: Int) -> None:
        _positive_number(num, "alm.max_iter")
        if not isinstance(num, int):
            raise TypeError("Alpaqa option 'alm.max_iter' must be an integer.")
        self.set_alm_option("max_iter", num)

    def set_panoc_maximum_iterations(self, num: Int) -> None:
        self.set_maximum_iterations(num)

    def set_initial_penalty(self, value: Float) -> None:
        _positive_number(value, "alm.initial_penalty", allow_zero=True)
        self.set_alm_option("initial_penalty", value)

    def set_penalty_update_factor(self, value: Float) -> None:
        _positive_number(value, "alm.penalty_update_factor")
        self.set_alm_option("penalty_update_factor", value)

    def set_maximum_penalty(self, value: Float) -> None:
        _positive_number(value, "alm.max_penalty")
        self.set_alm_option("max_penalty", value)

    def set_lbfgs_memory(self, value: Int) -> None:
        _positive_number(value, "lbfgs.memory")
        if not isinstance(value, int):
            raise TypeError("Alpaqa option 'lbfgs.memory' must be an integer.")
        self.set_lbfgs_option("memory", value)

    def set_dual_tolerance(self, value: Float) -> None:
        self.set_constraint_tolerance(value)

    def set_initial_tolerance(self, value: Float) -> None:
        _positive_number(value, "alm.initial_tolerance")
        self.set_alm_option("initial_tolerance", value)

    def set_maximum_wall_time(self, seconds: Float) -> None:
        _positive_number(seconds, "alm.max_time")
        self.set_alm_option("max_time", f"{seconds}s")

    def set_print_level(self, num: Int) -> None:
        if not isinstance(num, int):
            raise TypeError("Alpaqa print level must be an integer.")
        if num < 0:
            raise ValueError("Alpaqa print level must be non-negative.")
        interval = 0 if num == 0 else 1
        self.set_alm_option("print_interval", interval)
        self.set_panoc_option("print_interval", interval)

    def set_c_compile(self, val: bool) -> None:
        if val:
            raise NotImplementedError(
                "C compilation is not supported by the current CasADi alpaqa plugin."
            )
        self._c_compile = False

    def as_dict(self, solver) -> dict:
        nested = {
            group: values.copy()
            for group, values in self._alpaqa_options.items()
            if values
        }
        return {"alpaqa": nested, **solver.options_common}
