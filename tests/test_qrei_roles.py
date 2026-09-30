import pytest

from bearing_dt.qrei.roles import fold_roles

BEARINGS = ["B02", "B03", "B04", "B08", "B10", "B11", "B12", "B17"]


@pytest.mark.parametrize("roles", ["forward", "swap", "reverse"])
def test_roles_are_disjoint_and_complete(roles):
    for test in BEARINGS:
        train, validation, calibration = fold_roles(BEARINGS, test, roles)
        assert len({test, validation, calibration}) == 3
        assert sorted(train + [test, validation, calibration]) == sorted(BEARINGS)


def test_primary_roles_unchanged_and_swap_keeps_fitting_bearings():
    assert fold_roles(BEARINGS, "B03") == (["B02", "B10", "B11", "B12", "B17"], "B04", "B08")
    for test in BEARINGS:
        forward = fold_roles(BEARINGS, test)
        swap = fold_roles(BEARINGS, test, "swap")
        assert forward[0] == swap[0]
        assert (forward[1], forward[2]) == (swap[2], swap[1])
    assert fold_roles(BEARINGS, "B03", "reverse") == (["B04", "B08", "B10", "B11", "B12"], "B02", "B17")
