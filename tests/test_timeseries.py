"""Tests for the shared time-series helpers."""

import pytest

from financial_news.timeseries import winsorize

# analysis.py caps one tail (article counts); price.py caps both (returns).
BOTH_MODES = pytest.mark.parametrize("two_sided", [False, True])

# median = 1.0, MAD = 1.0
_BASELINE = [0.0] * 10 + [1.0] * 10 + [2.0] * 10


@BOTH_MODES
def test_winsorize_empty_list_returns_empty(two_sided):
    assert winsorize([], k=3.5, two_sided=two_sided) == []


@BOTH_MODES
def test_winsorize_zero_mad_returns_unchanged(two_sided):
    """Sparse baselines (majority sharing the median) skip capping entirely."""
    values = [0.0, 0.0, 0.0, 5.0]
    assert winsorize(values, k=3.5, two_sided=two_sided) == values


@BOTH_MODES
def test_winsorize_no_op_when_all_within_cap(two_sided):
    values = [1.0, 2.0, 1.0, 3.0, 2.0]
    assert winsorize(values, k=10.0, two_sided=two_sided) == values


@BOTH_MODES
@pytest.mark.parametrize("k, cap", [(3.5, 4.5), (1.0, 2.0)])
def test_winsorize_caps_high_outlier(two_sided, k, cap):
    result = winsorize(_BASELINE + [50.0], k=k, two_sided=two_sided)
    assert result[:-1] == _BASELINE
    assert result[-1] == pytest.approx(cap)


@pytest.mark.parametrize("two_sided, expected", [(False, -50.0), (True, -2.5)])
def test_winsorize_caps_low_outlier_only_when_two_sided(two_sided, expected):
    result = winsorize(_BASELINE + [-50.0], k=3.5, two_sided=two_sided)
    assert result[:-1] == _BASELINE
    assert result[-1] == pytest.approx(expected)
