import math

import pytest

from financial_news import analysis


def test_mean_zero_and_recent_zero():
    assert analysis.calculate_z_score(0, 0.0, 0.0) == 0.0


def test_mean_zero_and_recent_nonzero():
    z = analysis.calculate_z_score(5, 0.0, 0.0)
    assert math.isinf(z) and z > 0


def test_std_zero_returns_ratio():
    # recent_count / mean
    z = analysis.calculate_z_score(6, 2.0, 0.0)
    assert z == 3.0


def test_normal_case():
    # (recent - mean) / std
    z = analysis.calculate_z_score(6, 2.0, 2.0)
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
    z = analysis.calculate_z_score(recent, mean, std)
    if math.isinf(expected):
        assert math.isinf(z) and z > 0
    else:
        assert z == expected


# ── _winsorize_upper ────────────────────────────────────────────────────────


def test_winsorize_upper_empty_list_returns_empty():
    assert analysis._winsorize_upper([], k=3.5) == []


def test_winsorize_upper_zero_mad_returns_unchanged():
    """Sparse baselines (majority sharing the median) skip capping entirely."""
    values = [0.0, 0.0, 0.0, 5.0]
    assert analysis._winsorize_upper(values, k=3.5) == values


def test_winsorize_upper_no_op_when_all_within_cap():
    values = [1.0, 2.0, 1.0, 3.0, 2.0]
    assert analysis._winsorize_upper(values, k=10.0) == values


def test_winsorize_upper_caps_high_outlier():
    # median=1.0, MAD=1.0 -> cap = 1.0 + 3.5*1.0 = 4.5
    values = [0.0] * 10 + [1.0] * 10 + [50.0]
    result = analysis._winsorize_upper(values, k=3.5)
    assert result[:-1] == values[:-1]
    assert result[-1] == pytest.approx(4.5)


def test_winsorize_upper_smaller_k_caps_more_aggressively():
    values = [0.0] * 10 + [1.0] * 10 + [50.0]
    result = analysis._winsorize_upper(values, k=1.0)
    # cap = 1.0 + 1.0*1.0 = 2.0
    assert result[-1] == pytest.approx(2.0)
