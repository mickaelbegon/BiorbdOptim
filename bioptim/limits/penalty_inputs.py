"""Resolution of the node-dependent inputs passed to penalty functions.

This module deliberately depends only on :mod:`penalty_helpers`.  The callers
provide the storage-specific callbacks, so the same resolution rules can be
used while constructing CasADi functions, assembling the NLP, evaluating a
solution, and rendering plots.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .penalty_helpers import PenaltyHelpers

PenaltyDataGetter = Callable[[int, int, Any], Any]


@dataclass(frozen=True)
class PenaltyInputProvider:
    """Storage-specific callbacks used to resolve a penalty input at one node."""

    time: Callable[[int, int], Any]
    parameters: PenaltyDataGetter
    states: PenaltyDataGetter | None = None
    controls: PenaltyDataGetter | None = None
    algebraic_states: PenaltyDataGetter | None = None
    numerical_timeseries: PenaltyDataGetter | None = None


@dataclass(frozen=True)
class PenaltyConstructionContext:
    """The active controller and resolved position of a symbolic penalty."""

    controller: Any
    penalty_idx: int


@dataclass(frozen=True)
class PenaltyFunctionInputs:
    """Inputs that vary with a penalty node.

    ``phases_dt`` is intentionally supplied when formatting the arguments:
    it is shared by every node of a penalty and is therefore resolved once by
    the caller.
    """

    t0: Any
    x: Any
    u: Any
    p: Any
    a: Any
    d: Any
    weight: Any | None = None
    target: Any | None = None

    def function_arguments(self, phases_dt: Any) -> tuple[Any, ...]:
        """Return the arguments in the order expected by a CasADi penalty function."""

        return self.t0, phases_dt, self.x, self.u, self.p, self.a, self.d

    def weighted_function_arguments(self, phases_dt: Any) -> tuple[Any, ...]:
        """Return the arguments in the order expected by a weighted penalty function."""

        if self.weight is None or self.target is None:
            raise RuntimeError("weight and target must be resolved for a weighted penalty function")
        return *self.function_arguments(phases_dt), self.weight, self.target


class PenaltyInputResolver:
    """Resolve the inputs of one penalty node from caller-provided storage."""

    @staticmethod
    def resolve(
        penalty,
        penalty_idx: int,
        provider: PenaltyInputProvider,
        *,
        is_constructing_penalty: bool = False,
        include_weight_and_target: bool = False,
    ) -> PenaltyFunctionInputs:
        """Resolve all node-dependent penalty inputs.

        The selection rules remain in :class:`PenaltyHelpers` for now.  This
        first extraction centralizes their use without changing the public
        penalty API or the custom-function controller contract.
        """

        x = (
            PenaltyHelpers.states(
                penalty,
                penalty_idx,
                provider.states,
                is_constructing_penalty=is_constructing_penalty,
            )
            if provider.states is not None
            else []
        )
        u = (
            PenaltyHelpers.controls(
                penalty,
                penalty_idx,
                provider.controls,
                is_constructing_penalty=is_constructing_penalty,
            )
            if provider.controls is not None
            else []
        )
        a = (
            PenaltyHelpers.states(
                penalty,
                penalty_idx,
                provider.algebraic_states,
                is_constructing_penalty=is_constructing_penalty,
            )
            if provider.algebraic_states is not None
            else []
        )
        d = (
            PenaltyHelpers.numerical_timeseries(penalty, penalty_idx, provider.numerical_timeseries)
            if provider.numerical_timeseries is not None
            else []
        )

        return PenaltyFunctionInputs(
            t0=PenaltyHelpers.t0(penalty, penalty_idx, provider.time),
            x=x,
            u=u,
            p=PenaltyHelpers.parameters(penalty, penalty_idx, provider.parameters),
            a=a,
            d=d,
            weight=PenaltyHelpers.weight(penalty, penalty_idx) if include_weight_and_target else None,
            target=PenaltyHelpers.target(penalty, penalty_idx) if include_weight_and_target else None,
        )
