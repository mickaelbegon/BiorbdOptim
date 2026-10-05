from dataclasses import dataclass
from enum import Enum

from .weight import ConstraintWeight, ObjectiveWeight
from ..misc.enums import PenaltyType


class PenaltyNature(Enum):
    """The mathematical role of a penalty in the nonlinear program."""

    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"


@dataclass(frozen=True)
class PenaltyClassification:
    """The independent mathematical role and visibility of a penalty.

    ``PenaltyType`` predates the explicit objective/constraint distinction and only
    indicates whether a penalty is user-defined or internal. Keeping it as the
    origin preserves the public ``penalty_type`` API while making pool selection
    independent from the numerical value of a weight.
    """

    nature: PenaltyNature
    origin: PenaltyType

    def __post_init__(self):
        if self.origin not in (PenaltyType.USER, PenaltyType.INTERNAL):
            raise ValueError(f"Invalid penalty origin {self.origin}.")

    @classmethod
    def objective(cls, origin: PenaltyType) -> "PenaltyClassification":
        return cls(PenaltyNature.OBJECTIVE, origin)

    @classmethod
    def constraint(cls, origin: PenaltyType) -> "PenaltyClassification":
        return cls(PenaltyNature.CONSTRAINT, origin)

    @classmethod
    def from_weight(cls, weight, origin: PenaltyType) -> "PenaltyClassification":
        """Provide a compatibility bridge for direct ``PenaltyOption`` subclasses."""

        if isinstance(weight, ObjectiveWeight):
            return cls.objective(origin)
        if isinstance(weight, ConstraintWeight):
            return cls.constraint(origin)
        raise TypeError(f"Cannot infer a penalty nature from {type(weight)}.")

    def pool(self, owner):
        """Return the penalty pool on an OCP or NLP owner."""

        pool_name = {
            (PenaltyNature.OBJECTIVE, PenaltyType.USER): "J",
            (PenaltyNature.OBJECTIVE, PenaltyType.INTERNAL): "J_internal",
            (PenaltyNature.CONSTRAINT, PenaltyType.USER): "g",
            (PenaltyNature.CONSTRAINT, PenaltyType.INTERNAL): "g_internal",
        }[self.nature, self.origin]
        return getattr(owner, pool_name)
