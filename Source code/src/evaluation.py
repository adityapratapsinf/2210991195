"""
evaluation.py
=============
Evaluation metrics used in the paper:

    1. Exact Match (EM)         — for Q&A and code generation tasks
    2. ROUGE-L F1               — for summarisation tasks
    3. Composite Score P        — P = 0.40*A + 0.35*Q + 0.25*V
    4. Cohen's Kappa            — inter-rater reliability for human evaluation
    5. Normalise Likert         — converts 1–5 ratings to [0, 1]

Author  : Aditya Pratap Singh  (Roll: 2210991195)
Mentor  : Mr. Gurpreet Singh
Org     : Chitkara University, Punjab
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact Match
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> int:
    """
    Returns 1 if prediction matches reference exactly
    (case-insensitive, leading/trailing whitespace ignored), else 0.

    Used for: Open-domain Q&A, Code Generation (pass/fail).
    """
    return int(prediction.strip().lower() == reference.strip().lower())


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Average Exact Match over a list of predictions.

    Args:
        predictions : list of model output strings
        references  : list of ground-truth strings

    Returns:
        Float in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length.")
    scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUGE-L F1
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_length(x: list, y: list) -> int:
    """Compute the length of the Longest Common Subsequence (LCS)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(hypothesis: str, reference: str, beta: float = 1.0) -> float:
    """
    Compute ROUGE-L F1 between a hypothesis (model output) and reference.

    From the paper (Section III.G):
        R_L = LCS(H, R) / |R|
        P_L = LCS(H, R) / |H|
        F_L = (1 + β²) * R_L * P_L / (β² * P_L + R_L)

    β = 1.0 gives equal weight to precision and recall.

    Used for: Abstractive Summarisation.

    Args:
        hypothesis : model-generated summary
        reference  : ground-truth summary
        beta       : precision-recall balance (default 1.0)

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()

    if not h_tokens or not r_tokens:
        return 0.0

    lcs = _lcs_length(h_tokens, r_tokens)
    recall    = lcs / len(r_tokens)
    precision = lcs / len(h_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return round(f1, 4)


def rouge_l_score(hypotheses: list[str], references: list[str]) -> float:
    """
    Average ROUGE-L F1 over a list of hypothesis-reference pairs.

    Args:
        hypotheses : list of model-generated summaries
        references : list of ground-truth summaries

    Returns:
        Float in [0, 1].
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length.")
    scores = [rouge_l_f1(h, r) for h, r in zip(hypotheses, references)]
    return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composite Score
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    accuracy: float,
    reasoning_quality: float,
    clarity: float,
    alpha: float = 0.40,
    beta: float  = 0.35,
    gamma: float = 0.25,
) -> float:
    """
    Composite performance score from the paper (Section III.G):

        P_emr = α*A + β*Q + γ*V
              = 0.40*A + 0.35*Q + 0.25*V

    Weight rationale:
        α = 0.40  (Accuracy — most objectively verifiable)
        β = 0.35  (Reasoning Quality — strongest predictor of accuracy, r=0.97)
        γ = 0.25  (Clarity — usability of the response)

    All three input values must be in [0, 1].

    Args:
        accuracy          : normalised accuracy (EM or ROUGE-L F1)
        reasoning_quality : human-rated reasoning quality score
        clarity           : human-rated response clarity score
        alpha, beta, gamma: weights (must sum to exactly 1.0)

    Returns:
        Composite score in [0, 1].
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {alpha + beta + gamma:.4f}"
    return round(alpha * accuracy + beta * reasoning_quality + gamma * clarity, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Human Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_likert(score: int, scale: int = 5) -> float:
    """
    Convert a Likert scale integer to a [0, 1] float.

    Example:  score=4 on a 5-point scale → (4-1)/(5-1) = 0.75

    Args:
        score : integer rating (must be in [1, scale])
        scale : maximum rating value (default 5)

    Returns:
        Normalised score in [0, 1].
    """
    if not (1 <= score <= scale):
        raise ValueError(
            f"score must be between 1 and {scale}, got {score}."
        )
    return round((score - 1) / (scale - 1), 4)


def cohens_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """
    Compute Cohen's Kappa (κ) inter-rater reliability.

    From the paper (Section IV.C):
        Cases with κ < 0.60 were sent to a third evaluator for resolution.

    Formula:
        κ = (p_o - p_e) / (1 - p_e)
        where p_o = observed agreement, p_e = expected chance agreement

    Args:
        ratings_a : integer ratings from rater A
        ratings_b : integer ratings from rater B (same items)

    Returns:
        Cohen's kappa in [-1, 1].
        κ = 1.0  →  perfect agreement
        κ = 0.0  →  agreement no better than chance
        κ < 0    →  agreement worse than chance
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Both raters must rate the same number of items.")

    n = len(ratings_a)
    categories = list(set(ratings_a + ratings_b))

    p_o = sum(a == b for a, b in zip(ratings_a, ratings_b)) / n

    p_e = sum(
        (ratings_a.count(cat) / n) * (ratings_b.count(cat) / n)
        for cat in categories
    )

    if p_e == 1.0:
        return 1.0

    return round((p_o - p_e) / (1 - p_e), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Results Display
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict) -> None:
    """
    Print the results table matching Table I in the paper.

    Args:
        results : {
            technique_name: {
                'accuracy'  : float,
                'reasoning' : float,
                'clarity'   : float,
                'overall'   : float,
            }
        }
    """
    header = (
        f"{'Technique':<22} "
        f"{'Acc (A)':>9} "
        f"{'Reas (Q)':>9} "
        f"{'Clar (V)':>9} "
        f"{'Overall (P)':>12}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, s in results.items():
        print(
            f"{name:<22} "
            f"{s['accuracy']*100:>8.1f}% "
            f"{s['reasoning']*100:>8.1f}% "
            f"{s['clarity']*100:>8.1f}% "
            f"{s['overall']*100:>11.1f}%"
        )
    print(sep + "\n")
