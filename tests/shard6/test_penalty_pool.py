from bioptim.limits.penalty_pool import PenaltyPool


def test_reserve_slot_reuses_the_first_empty_slot():
    occupied_penalty = object()
    pool = [occupied_penalty, [], object()]

    index = PenaltyPool.reserve_slot(pool, -1)

    assert index == 1
    assert pool == [occupied_penalty, [], pool[2]]


def test_reserve_slot_appends_when_no_empty_slot_exists():
    pool = [object(), object()]

    index = PenaltyPool.reserve_slot(pool, -1)

    assert index == 2
    assert pool[-1] == []


def test_reserve_slot_extends_and_clears_an_explicit_slot():
    pool = [object()]

    index = PenaltyPool.reserve_slot(pool, 3)

    assert index == 3
    assert len(pool) == 4
    assert pool[1:] == [[], [], []]
