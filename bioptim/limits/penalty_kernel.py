from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PenaltyKernel:
    """Adapter around a legacy penalty implementation callable."""

    implementation: Callable[..., Any]

    def evaluate(self, penalty, controllers, **parameters):
        """Evaluate the kernel without changing the custom callback contract."""

        return self.implementation(penalty, controllers, **parameters)


class PenaltyKernelRegistry:
    """Resolve legacy ``FcnEnum`` members to evaluable penalty kernels.

    Penalty enums intentionally retain their historical ``(callable,)`` values.
    Future metadata or alternate implementations can be registered here without
    modifying those public enum definitions.
    """

    @staticmethod
    def resolve(function) -> PenaltyKernel:
        try:
            implementation = function.value[0]
        except (AttributeError, IndexError, TypeError) as error:
            raise TypeError(f"Invalid legacy penalty function {function!r}.") from error

        if not callable(implementation):
            raise TypeError(f"The legacy penalty function for {function!r} must be callable.")

        return PenaltyKernel(implementation)
