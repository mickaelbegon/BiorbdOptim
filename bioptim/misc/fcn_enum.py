from abc import abstractmethod
from enum import Enum

from ..limits.penalty_kernel import PenaltyKernelRegistry


class FcnEnum(Enum):
    def __call__(self, *args, **kwargs):
        """
        Call the member.
        """
        return self.kernel.evaluate(*args, **kwargs)

    @property
    def kernel(self):
        """Return the adapter for this legacy penalty function."""

        return PenaltyKernelRegistry.resolve(self)

    @staticmethod
    @abstractmethod
    def get_type():
        """
        Returns the type of the member.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_fcn_types():
        """
        Returns the types of the enum.
        """
        pass
