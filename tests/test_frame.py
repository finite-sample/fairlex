"""Tests for the data-frame front end."""

import numpy as np
import pandas as pd
import pytest

from fairlex import calibrate, calibrate_replicates
from fairlex.calibration import leximin_weights

ACC = 1e-5


@pytest.fixture
def survey():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "sex": rng.choice(["f", "m"], n),
            "age": rng.choice(["young", "mid", "old"], n, p=[0.2, 0.4, 0.4]),
            "w": rng.uniform(0.5, 2.0, n),
        },
        index=pd.RangeIndex(100, 100 + n),
    )


def test_matches_the_matrix_api(survey):
    targets = {
        "sex": {"f": 210.0, "m": 190.0},
        "age": {"young": 120.0, "mid": 150.0, "old": 130.0},
    }

    report = calibrate(survey, targets, base_weight="w", bounds=(0.5, 2.0))

    A = np.array(
        [survey.sex == "f", survey.sex == "m"]
        + [survey.age == a for a in ("young", "mid", "old")],
        dtype=float,
    )
    b = np.array([210.0, 190.0, 120.0, 150.0, 130.0])
    direct = leximin_weights(A, b, survey.w.to_numpy(), min_ratio=0.5, max_ratio=2.0)
    assert np.allclose(report.weights.to_numpy(), direct.w, atol=ACC)
    assert report.weights.index.equals(survey.index)
    assert np.allclose(report.margins["after"], A @ direct.w, atol=1e-3)


def test_conflicting_sources_are_named(survey):
    """Sex says 400 people, age says 460: every margin misses and conflicts."""
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 140.0, "mid": 160.0, "old": 160.0},
    }

    report = calibrate(survey, targets, base_weight="w")

    assert report.epsilon > 0.01
    assert {(v, lvl) for v, lvl, _ in report.binding} == {
        ("sex", "f"),
        ("sex", "m"),
        ("age", "young"),
        ("age", "mid"),
        ("age", "old"),
    }
    assert {why for _, _, why in report.binding} == {"conflicting targets"}


def test_unreachable_target_is_blamed_on_the_bounds(survey):
    """Young respondents would need 3x their weight; the cap is 2x."""
    young_now = survey.w[survey.age == "young"].sum()
    targets = {"age": {"young": 3 * young_now, "mid": 150.0, "old": 130.0}}

    report = calibrate(survey, targets, base_weight="w", bounds=(0.5, 2.0))

    assert ("age", "young", "weight bounds") in report.binding


def test_consistent_targets_report_no_conflicts(survey):
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 100.0, "mid": 150.0, "old": 150.0},
    }

    report = calibrate(survey, targets, base_weight="w")

    assert report.epsilon < ACC
    assert report.binding == []


def test_shares_resolve_disagreement_about_size(survey):
    targets = {
        "sex": {"f": 0.5, "m": 0.5},
        "age": {"young": 0.3, "mid": 0.35, "old": 0.35},
    }

    report = calibrate(survey, targets, shares=True, total=1000.0)

    assert report.epsilon < ACC
    assert np.isclose(report.weights.sum(), 1000.0, rtol=ACC)
    young = report.weights[survey.age == "young"].sum()
    assert np.isclose(young, 300.0, rtol=ACC)


def test_importance_protects_a_variable(survey):
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 140.0, "mid": 160.0, "old": 160.0},
    }

    plain = calibrate(survey, targets, base_weight="w")
    favoured = calibrate(survey, targets, base_weight="w", importance={"age": 4.0})

    age_miss = lambda r: r.margins.query("variable == 'age'")["miss_pct"].abs().max()  # noqa: E731
    assert age_miss(favoured) < age_miss(plain) / 2


def test_report_prints(survey):
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 140.0, "mid": 160.0, "old": 160.0},
    }

    text = str(calibrate(survey, targets, base_weight="w"))

    assert "worst-missed targets" in text
    assert "conflicting targets" in text
    assert "-0.0" not in text
    assert "young" in text


@pytest.mark.parametrize(
    ("targets", "kwargs", "match"),
    [
        ({"colour": {"red": 1.0}}, {}, "not a column"),
        ({"sex": {"f": 200.0}}, {}, "no target"),
        ({"sex": {"f": 1.0, "m": 1.0, "x": 1.0}}, {}, "no respondents"),
        ({"sex": {"f": 0.5, "m": 0.5}}, {"shares": True}, "total="),
        ({"sex": {"f": 1.0, "m": 1.0}}, {"importance": {"age": 2.0}}, "importance"),
        ({"sex": {"f": 1.0, "m": 1.0}}, {"importance": {"sex": 0.0}}, "importance"),
    ],
)
def test_bad_inputs_raise(survey, targets, kwargs, match):
    with pytest.raises(ValueError, match=match):
        calibrate(survey, targets, **kwargs)


def test_missing_values_raise(survey):
    survey.loc[survey.index[0], "sex"] = None
    with pytest.raises(ValueError, match="missing"):
        calibrate(survey, {"sex": {"f": 200.0, "m": 200.0}})


def test_replicates_match_calibrating_each_column(survey):
    rng = np.random.default_rng(5)
    for r in range(3):
        draws = rng.multinomial(len(survey), np.ones(len(survey)) / len(survey))
        survey[f"rep{r}"] = survey.w * draws
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 100.0, "mid": 150.0, "old": 150.0},
    }
    cols = ["rep0", "rep1", "rep2"]

    reps = calibrate_replicates(survey, targets, cols, bounds=(0.2, 5.0))

    assert list(reps.columns) == cols
    assert reps.index.equals(survey.index)
    for col in cols:
        one = calibrate(survey, targets, base_weight=col, bounds=(0.2, 5.0))
        assert np.allclose(reps[col], one.weights, atol=ACC)
        assert (reps[col][survey[col] == 0] == 0).all()


def test_replicates_refuse_base_weight(survey):
    with pytest.raises(ValueError, match="base_weight"):
        calibrate_replicates(
            survey, {"sex": {"f": 1.0, "m": 1.0}}, ["w"], base_weight="w"
        )


def test_zero_base_weights_do_not_warn(survey):
    """A bootstrap draw leaves some respondents at weight 0."""
    survey.loc[survey.index[:50], "w"] = 0.0
    targets = {
        "sex": {"f": 200.0, "m": 200.0},
        "age": {"young": 140.0, "mid": 160.0, "old": 160.0},
    }

    report = calibrate(survey, targets, base_weight="w")

    assert (report.weights.iloc[:50] == 0).all()
    assert report.binding


def test_slack_is_not_reported_as_a_conflict():
    """A miss that slack allows is not a binding target."""
    df = pd.DataFrame({"g": ["a"], "w": [1.0]})

    report = calibrate(df, {"g": {"a": 2.0}}, base_weight="w", slack=0.25)

    assert report.epsilon > 0.2
    assert report.binding == []


def test_missing_total_with_disagreeing_sizes_raises():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    targets = {"a": {"x": 1.0, "y": 1.0}, "b": {"p": 3.0, "q": 3.0}}

    with pytest.raises(ValueError, match="total="):
        calibrate(df, targets)


def test_missing_total_with_one_implied_size_is_fine():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    targets = {"a": {"x": 1.0, "y": 3.0}, "b": {"p": 1.0, "q": 3.0}}

    report = calibrate(df, targets)

    assert np.isclose(report.weights.sum(), 4.0, rtol=ACC)


def test_shares_must_sum_to_one(survey):
    with pytest.raises(ValueError, match="sum to 1"):
        calibrate(survey, {"sex": {"f": 0.2, "m": 0.2}}, shares=True, total=400.0)


def test_negative_targets_raise(survey):
    with pytest.raises(ValueError, match="non-negative"):
        calibrate(survey, {"sex": {"f": -1.0, "m": 401.0}})
