"""
tests/test_evaluation.py
=========================
Unit tests for the evaluation metrics module.
Run with:  python -m pytest tests/ -v

Author : Aditya Pratap Singh (Roll: 2210991195)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import (
    exact_match,
    exact_match_score,
    rouge_l_f1,
    rouge_l_score,
    composite_score,
    normalise_likert,
    cohens_kappa,
)


# ─── Exact Match ─────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") == 1

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1

    def test_mismatch(self):
        assert exact_match("London", "Paris") == 0

    def test_strips_whitespace(self):
        assert exact_match("  Paris  ", "Paris") == 1

    def test_empty_strings(self):
        assert exact_match("", "") == 1

    def test_score_over_list(self):
        preds = ["Paris", "Berlin", "London"]
        refs  = ["Paris", "Berlin", "Tokyo"]
        # 2 out of 3 correct
        assert abs(exact_match_score(preds, refs) - 2/3) < 1e-6


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────

class TestRougeL:
    def test_perfect_match(self):
        score = rouge_l_f1("the cat sat on the mat", "the cat sat on the mat")
        assert abs(score - 1.0) < 1e-4

    def test_no_overlap(self):
        score = rouge_l_f1("dog runs fast", "cat sat mat")
        assert score == 0.0

    def test_partial_overlap(self):
        score = rouge_l_f1("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        assert rouge_l_f1("", "reference text") == 0.0

    def test_empty_reference(self):
        assert rouge_l_f1("hypothesis text", "") == 0.0

    def test_score_over_list(self):
        hyps = ["the cat sat on the mat", "dogs run fast"]
        refs = ["the cat sat on the mat", "cats walk slowly"]
        avg = rouge_l_score(hyps, refs)
        assert 0.0 < avg < 1.0


# ─── Composite Score ──────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_paper_structured_score(self):
        # Paper Table I: Structured → A=0.91, Q=0.90, V=0.92, P=0.91
        p = composite_score(accuracy=0.91, reasoning_quality=0.90, clarity=0.92)
        assert abs(p - 0.909) < 0.001

    def test_paper_zero_shot_score(self):
        # Paper Table I: Zero-Shot → A=0.68, Q=0.64, V=0.70, P≈0.671
        p = composite_score(accuracy=0.68, reasoning_quality=0.64, clarity=0.70)
        assert abs(p - 0.671) < 0.001

    def test_all_zeros(self):
        assert composite_score(0, 0, 0) == 0.0

    def test_all_ones(self):
        assert composite_score(1, 1, 1) == 1.0

    def test_weights_must_sum_to_one(self):
        import pytest
        with pytest.raises(AssertionError):
            composite_score(0.5, 0.5, 0.5, alpha=0.5, beta=0.5, gamma=0.5)


# ─── Human Eval Helpers ───────────────────────────────────────────────────────

class TestNormaliseLikert:
    def test_score_1_on_5_point(self):
        assert normalise_likert(1, 5) == 0.0

    def test_score_5_on_5_point(self):
        assert normalise_likert(5, 5) == 1.0

    def test_score_3_on_5_point(self):
        assert abs(normalise_likert(3, 5) - 0.5) < 1e-4

    def test_invalid_score_raises(self):
        import pytest
        with pytest.raises(ValueError):
            normalise_likert(6, 5)


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        assert cohens_kappa(a, b) == 1.0

    def test_below_threshold(self):
        # Completely random disagreement → kappa near 0 or negative
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        kappa = cohens_kappa(a, b)
        assert kappa < 0.6

    def test_mismatched_lengths_raises(self):
        import pytest
        with pytest.raises(ValueError):
            cohens_kappa([1, 2], [1, 2, 3])
