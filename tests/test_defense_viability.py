from __future__ import annotations

import pandas as pd
import pytest

from domain_defenses.mcq_analysis import defense_viability_diagnostics


def _make_df(initial_scores: list[int], post_scores: list[int]) -> pd.DataFrame:
    """Build a minimal paired DataFrame for N questions."""
    assert len(initial_scores) == len(post_scores)
    rows = []
    for i, (init_s, post_s) in enumerate(zip(initial_scores, post_scores)):
        qid = f"q{i}"
        rows.append({"id": qid, "phase": "initial", "score": float(init_s), "parse_failed": False})
        rows.append({"id": qid, "phase": "post_context", "score": float(post_s), "parse_failed": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# viable path
# ---------------------------------------------------------------------------


def test_viable_when_conditions_met() -> None:
    # 25 questions all initially correct, 10 of them flip → susceptibility 40%
    n = 25
    diag = defense_viability_diagnostics(_make_df([1] * n, [0] * 10 + [1] * 15))
    assert diag["viable"] is True
    assert diag["warnings"] == []


def test_viable_counts_and_rates_are_correct() -> None:
    # 30 initially correct, 6 flip  →  susceptibility = 6/30 = 20%
    n = 30
    diag = defense_viability_diagnostics(_make_df([1] * n, [0] * 6 + [1] * 24))
    assert diag["n_initial"] == n
    assert diag["n_attackable"] == n
    assert diag["n_flipped_c2i"] == 6
    assert abs(diag["susceptibility_rate"] - 6 / 30) < 1e-9
    assert abs(diag["attackable_fraction"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# not viable — small attackable pool
# ---------------------------------------------------------------------------


def test_not_viable_when_attackable_pool_too_small() -> None:
    # Only 5 initially-correct questions — below MIN_ATTACKABLE=20
    diag = defense_viability_diagnostics(_make_df([1] * 5, [0] * 5))
    assert diag["viable"] is False
    assert any("too small" in w for w in diag["warnings"])


# ---------------------------------------------------------------------------
# not viable — low initial accuracy
# ---------------------------------------------------------------------------


def test_not_viable_when_initial_accuracy_too_low() -> None:
    # 100 total, 20 initially correct (20% < 30%), 10 of those flip
    initial = [1] * 20 + [0] * 80
    post = [0] * 10 + [1] * 10 + [0] * 80
    diag = defense_viability_diagnostics(_make_df(initial, post))
    assert diag["viable"] is False
    assert any("accuracy" in w.lower() for w in diag["warnings"])


def test_attackable_fraction_matches_initial_accuracy() -> None:
    # 40 correct out of 100 = 40% initial accuracy
    initial = [1] * 40 + [0] * 60
    post = [0] * 10 + [1] * 30 + [0] * 60
    diag = defense_viability_diagnostics(_make_df(initial, post))
    assert abs(diag["attackable_fraction"] - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# not viable — baseline not susceptible
# ---------------------------------------------------------------------------


def test_not_viable_when_baseline_not_susceptible() -> None:
    # 30 initially correct, none flip → susceptibility = 0%
    n = 30
    diag = defense_viability_diagnostics(_make_df([1] * n, [1] * n))
    assert diag["viable"] is False
    assert any("susceptibility" in w.lower() for w in diag["warnings"])


def test_not_viable_when_susceptibility_below_threshold() -> None:
    # 30 correct, only 1 flip → susceptibility = 3.3% < 5%
    diag = defense_viability_diagnostics(_make_df([1] * 30, [0] * 1 + [1] * 29))
    assert diag["viable"] is False
    assert any("susceptibility" in w.lower() for w in diag["warnings"])


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_df_returns_not_viable() -> None:
    empty = pd.DataFrame(columns=["id", "phase", "score", "parse_failed"])
    diag = defense_viability_diagnostics(empty)
    assert diag["viable"] is False
    assert diag["n_initial"] == 0
    assert len(diag["warnings"]) > 0


def test_susceptibility_zero_when_no_attackable_questions() -> None:
    # All initially wrong, so n_attackable = 0
    diag = defense_viability_diagnostics(_make_df([0] * 30, [1] * 30))
    assert diag["n_attackable"] == 0
    assert diag["susceptibility_rate"] == 0.0
