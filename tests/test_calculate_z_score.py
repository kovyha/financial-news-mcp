import math

import pytest

from financial_news import server


def test_mean_zero_and_recent_zero():
    assert server.calculate_z_score(0, 0.0, 0.0) == 0.0


def test_mean_zero_and_recent_nonzero():
    z = server.calculate_z_score(5, 0.0, 0.0)
    assert math.isinf(z) and z > 0


def test_std_zero_returns_ratio():
    # recent_count / mean
    z = server.calculate_z_score(6, 2.0, 0.0)
    assert z == 3.0


def test_normal_case():
    # (recent - mean) / std
    z = server.calculate_z_score(6, 2.0, 2.0)
    assert z == 2.0


@pytest.mark.parametrize(
    "recent,mean,std,expected",
    [
        (1, 1.0, 0.0, 1.0),  # std=0 -> recent/mean
        (10, 1.0, 0.0, 10.0),  # large spike with zero std
        (0, 0.0, 0.0, 0.0),  # mean==0 and recent==0
        (5, 0.0, 0.0, float("inf")),  # mean==0 and recent>0 -> inf
        (6, 2.0, 2.0, 2.0),  # normal formula
        (3, 2.0, 0.5, 2.0),  # (3-2)/0.5 = 2
    ],
)
def test_calculate_z_score_parametrized(recent, mean, std, expected):
    z = server.calculate_z_score(recent, mean, std)
    if math.isinf(expected):
        assert math.isinf(z) and z > 0
    else:
        assert z == expected
