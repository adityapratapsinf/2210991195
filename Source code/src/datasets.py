"""
datasets.py
===========
Dataset loaders for the four benchmark task types used in the paper:

    1. Open-Domain Q&A        (TriviaQA format)
    2. Mathematical Reasoning (GSM8K format)
    3. Code Generation        (HumanEval format)
    4. Abstractive Summary    (CNN/DailyMail format)

Each loader accepts an optional JSONL file path.
If no file is given, it returns built-in sample items so the code
runs out of the box without downloading any datasets.

Author  : Aditya Pratap Singh  (Roll: 2210991195)
Mentor  : Mr. Gurpreet Singh
Org     : Chitkara University, Punjab
"""

import json
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Built-in sample items
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QA: list[dict] = [
    {"question": "What is the capital of France?",                    "answer": "Paris"},
    {"question": "Who wrote the play Hamlet?",                        "answer": "William Shakespeare"},
    {"question": "What is the chemical symbol for gold?",             "answer": "Au"},
    {"question": "In which year did World War II end?",               "answer": "1945"},
    {"question": "Which planet is known as the Red Planet?",          "answer": "Mars"},
    {"question": "What is the speed of light in km/s?",              "answer": "299792"},
    {"question": "Who developed the theory of general relativity?",   "answer": "Albert Einstein"},
    {"question": "What is the largest ocean on Earth?",               "answer": "Pacific Ocean"},
]

SAMPLE_MATH: list[dict] = [
    {
        "question": (
            "Sarah has 5 apples. She buys 3 more bags, each with 4 apples. "
            "She gives 6 apples to her friends. How many apples does she have left?"
        ),
        "answer": "11",
        "reasoning": (
            "Start: 5 apples. "
            "Buys 3 × 4 = 12 more. Total = 5 + 12 = 17. "
            "Gives away 6. Remaining = 17 - 6 = 11."
        ),
    },
    {
        "question": "A train travels at 80 km/h for 2.5 hours. How far does it travel?",
        "answer": "200 km",
        "reasoning": "Distance = speed × time = 80 × 2.5 = 200 km.",
    },
    {
        "question": (
            "A shirt originally costs $60. It is on a 30% discount. "
            "What is the final price after the discount?"
        ),
        "answer": "$42",
        "reasoning": "Discount = 30% of $60 = $18. Final price = $60 - $18 = $42.",
    },
    {
        "question": (
            "There are 24 students in a class. "
            "If 1/3 of them play football and 1/4 play cricket, "
            "how many play neither sport?"
        ),
        "answer": "10",
        "reasoning": (
            "Football: 24 / 3 = 8. Cricket: 24 / 4 = 6. "
            "Play at least one sport = 8 + 6 = 14 (assuming no overlap). "
            "Neither = 24 - 14 = 10."
        ),
    },
]

SAMPLE_CODE: list[dict] = [
    {
        "prompt": (
            "def sum_list(numbers: list[int]) -> int:\n"
            '    """Return the sum of all integers in the list.\n'
            "    Returns 0 for an empty list.\n"
            '    """\n'
        ),
        "canonical_solution": (
            "def sum_list(numbers: list[int]) -> int:\n"
            '    """Return the sum of all integers in the list.\n'
            "    Returns 0 for an empty list.\n"
            '    """\n'
            "    return sum(numbers)\n"
        ),
        "test": (
            "assert sum_list([1, 2, 3]) == 6\n"
            "assert sum_list([]) == 0\n"
            "assert sum_list([-1, 1]) == 0\n"
        ),
    },
    {
        "prompt": (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True if s reads the same forwards and backwards,\n'
            "    False otherwise. Comparison is case-sensitive.\n"
            '    """\n'
        ),
        "canonical_solution": (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True if s reads the same forwards and backwards,\n'
            "    False otherwise. Comparison is case-sensitive.\n"
            '    """\n'
            "    return s == s[::-1]\n"
        ),
        "test": (
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('A') == True\n"
        ),
    },
    {
        "prompt": (
            "def count_vowels(text: str) -> int:\n"
            '    """Return the number of vowels (a, e, i, o, u) in text.\n'
            "    Case-insensitive.\n"
            '    """\n'
        ),
        "canonical_solution": (
            "def count_vowels(text: str) -> int:\n"
            '    """Return the number of vowels (a, e, i, o, u) in text.\n'
            "    Case-insensitive.\n"
            '    """\n'
            "    return sum(1 for ch in text.lower() if ch in 'aeiou')\n"
        ),
        "test": (
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('AEIOU') == 5\n"
            "assert count_vowels('') == 0\n"
            "assert count_vowels('bcdfg') == 0\n"
        ),
    },
]

SAMPLE_SUMMARIES: list[dict] = [
    {
        "article": (
            "Scientists at CERN announced today that they have found evidence of a new "
            "subatomic particle using the Large Hadron Collider. The particle, tentatively "
            "named the X boson, was detected during experiments conducted last month. "
            "If confirmed, the discovery could significantly alter the standard model of "
            "particle physics. The research team plans to publish their full findings in "
            "Nature Physics next week and present at a conference in Geneva."
        ),
        "reference_summary": (
            "CERN scientists found evidence of a new subatomic particle called the X boson "
            "using the Large Hadron Collider, which could reshape the standard model of physics."
        ),
    },
    {
        "article": (
            "The government announced a new policy today that will provide free school meals "
            "to all primary school children across the country starting next academic term. "
            "The initiative is estimated to cost £500 million annually and is aimed at reducing "
            "inequality and improving educational outcomes. Opposition parties praised the move "
            "while also calling for it to be extended to secondary schools."
        ),
        "reference_summary": (
            "The government will provide free meals to all primary school children from next term, "
            "at an annual cost of £500 million, drawing broad cross-party support."
        ),
    },
    {
        "article": (
            "Researchers at MIT have developed a new battery technology that can charge a "
            "standard electric vehicle in under five minutes. The battery uses a novel "
            "solid-state electrolyte that allows for much faster ion transfer compared to "
            "conventional lithium-ion batteries. The team says the technology could be "
            "commercially available within three years if manufacturing challenges are resolved."
        ),
        "reference_summary": (
            "MIT researchers developed a solid-state battery that charges electric vehicles "
            "in under five minutes, potentially reaching the market within three years."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str | None, fallback: list[dict]) -> list[dict]:
    """Load a JSONL file if it exists, otherwise return built-in samples."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(f"  Loaded {len(data):,} items from {path}")
        return data
    print(f"  No file given — using {len(fallback)} built-in sample items.")
    return fallback


def load_qa(path: str | None = None) -> list[dict]:
    """
    Load open-domain Q&A data.

    Expected JSONL format (one object per line):
        {"question": "...", "answer": "..."}
    """
    return _load_jsonl(path, SAMPLE_QA)


def load_math(path: str | None = None) -> list[dict]:
    """
    Load GSM8K-style mathematical reasoning problems.

    Expected JSONL format:
        {"question": "...", "answer": "...", "reasoning": "..."}
    """
    return _load_jsonl(path, SAMPLE_MATH)


def load_code(path: str | None = None) -> list[dict]:
    """
    Load HumanEval-style Python code generation problems.

    Expected JSONL format:
        {"prompt": "...", "canonical_solution": "...", "test": "..."}
    """
    return _load_jsonl(path, SAMPLE_CODE)


def load_summaries(path: str | None = None) -> list[dict]:
    """
    Load CNN/DailyMail-style news summarisation items.

    Expected JSONL format:
        {"article": "...", "reference_summary": "..."}
    """
    return _load_jsonl(path, SAMPLE_SUMMARIES)
