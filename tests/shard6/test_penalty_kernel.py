import pytest

from bioptim import ObjectiveFcn
from bioptim.limits.penalty_kernel import PenaltyKernelRegistry
from bioptim.misc.fcn_enum import FcnEnum


def sentinel_kernel(penalty, controllers, token):
    return penalty, controllers, token


class SentinelPenaltyFcn(FcnEnum):
    SENTINEL = (sentinel_kernel,)

    @staticmethod
    def get_type():
        return None

    @staticmethod
    def get_fcn_types():
        return None


class InvalidPenaltyFcn(FcnEnum):
    INVALID = (None,)

    @staticmethod
    def get_type():
        return None

    @staticmethod
    def get_fcn_types():
        return None


def test_penalty_kernel_preserves_the_legacy_enum_value_contract():
    function = ObjectiveFcn.Mayer.MINIMIZE_TIME

    assert function.kernel.implementation is function.value[0]


def test_penalty_kernel_dispatches_enum_calls_and_direct_evaluation_identically():
    expected = ("penalty", "controllers", "sentinel")

    assert SentinelPenaltyFcn.SENTINEL("penalty", "controllers", token="sentinel") == expected
    assert SentinelPenaltyFcn.SENTINEL.kernel.evaluate("penalty", "controllers", token="sentinel") == expected


def test_penalty_kernel_rejects_invalid_legacy_values():
    with pytest.raises(TypeError, match="must be callable"):
        PenaltyKernelRegistry.resolve(InvalidPenaltyFcn.INVALID)
