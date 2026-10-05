import pytest

from bioptim import (
    Constraint,
    ConstraintFcn,
    ConstraintWeight,
    MultinodeConstraint,
    MultinodeConstraintFcn,
    MultinodeObjective,
    MultinodeObjectiveFcn,
    Node,
    Objective,
    ObjectiveFcn,
    ObjectiveWeight,
    PhaseTransition,
    PhaseTransitionFcn,
)
from bioptim.limits.penalty_classification import PenaltyClassification, PenaltyNature
from bioptim.limits.phase_transtion_factory import PhaseTransitionFactory
from bioptim.limits.constraints import ParameterConstraint
from bioptim.limits.objective_functions import ParameterObjective
from bioptim.misc.enums import InterpolationType, PenaltyType


class PenaltyPools:
    def __init__(self):
        self.J = []
        self.J_internal = []
        self.g = []
        self.g_internal = []


class PenaltyControllerStub:
    def __init__(self, ocp, nlp):
        self.ocp = ocp
        self.get_nlp = nlp


@pytest.mark.parametrize(
    "classification,pool_name",
    [
        (PenaltyClassification.objective(PenaltyType.USER), "J"),
        (PenaltyClassification.objective(PenaltyType.INTERNAL), "J_internal"),
        (PenaltyClassification.constraint(PenaltyType.USER), "g"),
        (PenaltyClassification.constraint(PenaltyType.INTERNAL), "g_internal"),
    ],
)
def test_penalty_classification_selects_the_expected_pool(classification, pool_name):
    pools = PenaltyPools()

    assert classification.pool(pools) is getattr(pools, pool_name)


@pytest.mark.parametrize(
    "weight,nature",
    [
        (ObjectiveWeight(0), PenaltyNature.OBJECTIVE),
        (ConstraintWeight(1), PenaltyNature.CONSTRAINT),
    ],
)
def test_penalty_classification_preserves_weight_type_compatibility(weight, nature):
    classification = PenaltyClassification.from_weight(weight, PenaltyType.USER)

    assert classification.nature is nature
    assert classification.origin is PenaltyType.USER


def test_objective_and_constraint_assign_their_nature_explicitly():
    objective = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME)
    constraint = Constraint(ConstraintFcn.TIME_CONSTRAINT)

    assert objective.classification == PenaltyClassification.objective(PenaltyType.USER)
    assert constraint.classification == PenaltyClassification.constraint(PenaltyType.USER)


@pytest.mark.parametrize(
    "penalty_factory,origin,pool_name",
    [
        (lambda origin: Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, penalty_type=origin), PenaltyType.USER, "J"),
        (
            lambda origin: Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, penalty_type=origin),
            PenaltyType.INTERNAL,
            "J_internal",
        ),
        (lambda origin: Constraint(ConstraintFcn.TIME_CONSTRAINT, penalty_type=origin), PenaltyType.USER, "g"),
        (
            lambda origin: Constraint(ConstraintFcn.TIME_CONSTRAINT, penalty_type=origin),
            PenaltyType.INTERNAL,
            "g_internal",
        ),
    ],
)
def test_penalty_option_routes_to_the_classified_phase_pool(penalty_factory, origin, pool_name):
    ocp = PenaltyPools()
    nlp = PenaltyPools()
    penalty = penalty_factory(origin)

    penalty.list_index = -1
    penalty.ensure_penalty_sanity(ocp, nlp)
    penalty._add_penalty_to_pool([PenaltyControllerStub(ocp, nlp)])

    assert getattr(nlp, pool_name) == [penalty]


@pytest.mark.parametrize(
    "penalty,pool_name",
    [
        (ParameterObjective(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER), "J"),
        (ParameterConstraint(ConstraintFcn.TIME_CONSTRAINT), "g"),
    ],
)
def test_parameter_penalty_routes_to_the_classified_ocp_pool(penalty, pool_name):
    ocp = PenaltyPools()
    nlp = PenaltyPools()

    penalty.list_index = -1
    penalty.ensure_penalty_sanity(ocp, nlp)
    penalty._add_penalty_to_pool([PenaltyControllerStub(ocp, nlp)])

    assert getattr(ocp, pool_name) == [penalty]
    assert getattr(nlp, pool_name) == []


def test_multinode_penalties_keep_an_internal_origin_with_explicit_natures():
    objective = MultinodeObjective(
        MultinodeObjectiveFcn,
        nodes=(Node.END, Node.START),
        nodes_phase=(0, 1),
        multinode_penalty=MultinodeObjectiveFcn.STATES_EQUALITY,
        weight=ObjectiveWeight(),
    )
    constraint = MultinodeConstraint(
        MultinodeConstraintFcn,
        nodes=(Node.END, Node.START),
        nodes_phase=(0, 1),
        multinode_penalty=MultinodeConstraintFcn.STATES_EQUALITY,
        weight=ConstraintWeight(),
    )

    assert objective.classification == PenaltyClassification.objective(PenaltyType.INTERNAL)
    assert constraint.classification == PenaltyClassification.constraint(PenaltyType.INTERNAL)
    assert objective.penalty_type is PenaltyType.INTERNAL
    assert constraint.penalty_type is PenaltyType.INTERNAL


@pytest.mark.parametrize(
    "weight,nature",
    [
        (ConstraintWeight(1), PenaltyNature.CONSTRAINT),
        (ObjectiveWeight(0), PenaltyNature.OBJECTIVE),
        (ObjectiveWeight([0, 1], interpolation=InterpolationType.EACH_FRAME), PenaltyNature.OBJECTIVE),
        (0, PenaltyNature.OBJECTIVE),
    ],
)
def test_phase_transition_classification_is_independent_from_weight_truthiness(weight, nature):
    transition = PhaseTransition(0, PhaseTransitionFcn.CONTINUOUS, weight=weight)

    assert transition.classification.nature is nature
    assert transition.classification.origin is PenaltyType.INTERNAL

    PhaseTransitionFactory.update_transition_base(None, transition)

    assert hasattr(transition, "base") is (nature is PenaltyNature.OBJECTIVE)
