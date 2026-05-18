"""
run_experiment.py
=================
End-to-end experiment script that evaluates all five prompting techniques
across all four benchmark task types and prints a results table matching
Table I of the paper.

Usage
-----
# Run with built-in sample data (no dataset files needed):
    python run_experiment.py

# Run with your own dataset files:
    python run_experiment.py \\
        --qa   data/qa.jsonl \\
        --math data/math.jsonl \\
        --code data/code.jsonl \\
        --summ data/summaries.jsonl

# See the reference results from the paper without making any API calls:
    python run_experiment.py --paper-results

Before running, make sure your API key is set:
    export OPENAI_API_KEY="sk-..."   (Linux / macOS)
    set    OPENAI_API_KEY=sk-...     (Windows CMD)

Author  : Aditya Pratap Singh  (Roll: 2210991195)
Mentor  : Mr. Gurpreet Singh
Org     : Chitkara University, Punjab
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv          # pip install python-dotenv
load_dotenv()                           # loads .env file if present

from prompting_techniques import (
    zero_shot, few_shot, chain_of_thought, role_based, structured,
)
from evaluation import (
    exact_match, rouge_l_f1, composite_score, print_results_table,
)
from datasets import load_qa, load_math, load_code, load_summaries


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration examples  (used in Few-Shot and CoT tasks)
# ─────────────────────────────────────────────────────────────────────────────

QA_FEW_SHOT_EXAMPLES = [
    {"input": "What is the capital of Germany?",  "output": "Berlin"},
    {"input": "Who invented the telephone?",       "output": "Alexander Graham Bell"},
    {"input": "What is the boiling point of water in Celsius?", "output": "100"},
]

MATH_COT_EXAMPLES = [
    {
        "input":     "Tom has 12 oranges and gives 4 to his sister. How many are left?",
        "reasoning": "Tom starts with 12. He gives away 4. 12 - 4 = 8.",
        "output":    "8",
    },
    {
        "input":     "A car travels at 80 km/h for 3 hours. How far does it go?",
        "reasoning": "Distance = speed × time = 80 × 3 = 240 km.",
        "output":    "240 km",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Paper reference results (Table I)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_RESULTS = {
    "Zero-Shot":        {"accuracy": 0.68, "reasoning": 0.64, "clarity": 0.70, "overall": 0.671},
    "Few-Shot":         {"accuracy": 0.78, "reasoning": 0.75, "clarity": 0.80, "overall": 0.777},
    "Role-Based":       {"accuracy": 0.82, "reasoning": 0.80, "clarity": 0.84, "overall": 0.822},
    "Chain-of-Thought": {"accuracy": 0.86, "reasoning": 0.89, "clarity": 0.85, "overall": 0.869},
    "Structured":       {"accuracy": 0.91, "reasoning": 0.90, "clarity": 0.92, "overall": 0.909},
}


# ─────────────────────────────────────────────────────────────────────────────
# Task runners — one per task type, one per technique
# ─────────────────────────────────────────────────────────────────────────────

def _pause():
    """Small pause between API calls to avoid hitting rate limits."""
    time.sleep(0.8)


def run_qa(technique: str, data: list[dict]) -> float:
    scores = []
    for item in data:
        q, ref = item["question"], item["answer"]

        if technique == "zero_shot":
            pred = zero_shot(f"Answer in one word or short phrase only:\n{q}")

        elif technique == "few_shot":
            pred = few_shot(f"Answer in one word or short phrase:\n{q}", QA_FEW_SHOT_EXAMPLES)

        elif technique == "cot":
            pred = chain_of_thought(f"Answer this question:\n{q}")

        elif technique == "role_based":
            pred = role_based(
                f"Answer in one word or short phrase:\n{q}",
                persona="You are a knowledgeable general knowledge expert.",
            )

        else:  # structured
            pred = structured(
                task_steps=[
                    "Read the question carefully.",
                    "Recall the correct factual answer.",
                    "State the answer in one word or short phrase.",
                ],
                persona="You are an expert in general knowledge and trivia.",
                constraints=["No explanation — answer only.", "Keep it short."],
                output_format="A single word or short phrase.",
            )

        scores.append(exact_match(pred, ref))
        _pause()

    return sum(scores) / len(scores) if scores else 0.0


def run_math(technique: str, data: list[dict]) -> float:
    scores = []
    for item in data:
        q, ref = item["question"], item["answer"]

        if technique == "zero_shot":
            pred = zero_shot(f"Solve and give only the final answer:\n{q}")

        elif technique == "few_shot":
            examples = [{"input": e["input"], "output": e["output"]} for e in MATH_COT_EXAMPLES]
            pred = few_shot(q, examples)

        elif technique == "cot":
            pred = chain_of_thought(q, few_shot_examples=MATH_COT_EXAMPLES)

        elif technique == "role_based":
            pred = role_based(
                f"Solve this maths problem:\n{q}",
                persona="You are an expert mathematics teacher who shows clear working.",
            )

        else:  # structured
            pred = structured(
                task_steps=[
                    "List all quantities given in the problem.",
                    "Identify what needs to be found.",
                    "Show your working step by step.",
                    "State the final answer with units.",
                ],
                persona="You are an expert mathematician.",
                constraints=["Show all working steps.", "Include units in the final answer."],
                output_format="Step-by-step working, then 'Final Answer: <value>'.",
            )

        # extract the number after "Final Answer:" if structured format was used
        answer_text = pred.split("Final Answer:")[-1].strip() if "Final Answer:" in pred else pred
        scores.append(exact_match(answer_text, ref))
        _pause()

    return sum(scores) / len(scores) if scores else 0.0


def run_summarisation(technique: str, data: list[dict]) -> float:
    scores = []
    for item in data:
        article, ref = item["article"], item["reference_summary"]

        if technique == "zero_shot":
            pred = zero_shot(f"Summarise the following in one sentence:\n\n{article}")

        elif technique == "few_shot":
            examples = [{
                "input":  "The government passed a new budget increasing education funding by 10%.",
                "output": "The government raised education spending by 10% in its latest budget.",
            }]
            pred = few_shot(f"Summarise in one sentence:\n\n{article}", examples)

        elif technique == "cot":
            pred = chain_of_thought(f"Summarise this article in one sentence:\n\n{article}")

        elif technique == "role_based":
            pred = role_based(
                f"Summarise the following article in one sentence:\n\n{article}",
                persona="You are a professional news editor who writes concise, accurate summaries.",
            )

        else:  # structured
            pred = structured(
                task_steps=[
                    "Identify the main subject and key event.",
                    "Note the single most important detail or figure.",
                    "Write one clear summary sentence.",
                ],
                persona="You are a professional journalist.",
                constraints=[
                    "Exactly one sentence.",
                    "Do not copy sentences from the article verbatim.",
                    "Include the most newsworthy fact.",
                ],
                output_format="A single sentence summary.",
            )

        scores.append(rouge_l_f1(pred, ref))
        _pause()

    return sum(scores) / len(scores) if scores else 0.0


def run_code(technique: str, data: list[dict]) -> float:
    scores = []
    for item in data:
        fn_prompt = item["prompt"]
        test_code = item["test"]

        if technique == "zero_shot":
            pred = zero_shot(f"Complete this Python function:\n\n{fn_prompt}")

        elif technique == "few_shot":
            example_fn = (
                "def add(a: int, b: int) -> int:\n"
                '    """Return a + b."""\n'
                "    return a + b\n"
            )
            examples = [{"input": "Complete:\ndef add(a, b):\n    ...", "output": example_fn}]
            pred = few_shot(f"Complete this Python function:\n\n{fn_prompt}", examples)

        elif technique == "cot":
            pred = chain_of_thought(
                f"Think through edge cases, then complete this Python function:\n\n{fn_prompt}"
            )

        elif technique == "role_based":
            pred = role_based(
                f"Complete this Python function:\n\n{fn_prompt}",
                persona=(
                    "You are a senior Python software engineer with 10 years of experience "
                    "writing clean, well-tested code."
                ),
            )

        else:  # structured
            pred = structured(
                task_steps=[
                    "Read the function signature and docstring carefully.",
                    "List all edge cases (empty input, None, zero, negative values).",
                    "Write the complete implementation with type annotations.",
                    "Add an inline comment for each key step.",
                ],
                persona="You are a senior Python engineer who writes clean, well-tested code.",
                constraints=[
                    "No third-party libraries.",
                    "Handle all edge cases.",
                    "Include type annotations.",
                    "Return only the function — no extra text or markdown.",
                ],
                output_format="Complete Python function starting with 'def'.",
            )

        code = _extract_code(pred)
        scores.append(int(_run_tests(code, test_code)))
        _pause()

    return sum(scores) / len(scores) if scores else 0.0


def _extract_code(response: str) -> str:
    """Pull Python code out of a markdown fence if the model wrapped it."""
    if "```python" in response:
        return response.split("```python")[1].split("```")[0].strip()
    if "```" in response:
        return response.split("```")[1].split("```")[0].strip()
    return response.strip()


def _run_tests(code: str, tests: str) -> bool:
    """Execute generated code + unit tests in an isolated namespace."""
    try:
        ns: dict = {}
        exec(code, ns)
        exec(tests, ns)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

TECHNIQUES = {
    "zero_shot":  "Zero-Shot",
    "few_shot":   "Few-Shot",
    "cot":        "Chain-of-Thought",
    "role_based": "Role-Based",
    "structured": "Structured",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt Engineering Experiment Runner — Aditya Pratap Singh (2210991195)"
    )
    parser.add_argument("--qa",           default=None, help="Path to Q&A JSONL file")
    parser.add_argument("--math",         default=None, help="Path to maths JSONL file")
    parser.add_argument("--code",         default=None, help="Path to code JSONL file")
    parser.add_argument("--summ",         default=None, help="Path to summarisation JSONL file")
    parser.add_argument("--paper-results", action="store_true",
                        help="Print reference results from the paper (no API calls)")
    args = parser.parse_args()

    if args.paper_results:
        print("\n=== Reference Results from the Paper (Table I) ===")
        print_results_table(PAPER_RESULTS)
        return

    # ── Load datasets ────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    qa_data   = load_qa(args.qa)
    math_data = load_math(args.math)
    code_data = load_code(args.code)
    summ_data = load_summaries(args.summ)

    print(
        f"\nDataset sizes → QA: {len(qa_data)}  |  "
        f"Math: {len(math_data)}  |  "
        f"Code: {len(code_data)}  |  "
        f"Summarisation: {len(summ_data)}"
    )
    print("\nRunning experiments (this will make OpenAI API calls) …\n")

    experiment_results: dict = {}

    for tech_key, tech_label in TECHNIQUES.items():
        print(f"  [{tech_label}]")

        qa_acc   = run_qa(tech_key, qa_data)
        math_acc = run_math(tech_key, math_data)
        code_acc = run_code(tech_key, code_data)
        summ_acc = run_summarisation(tech_key, summ_data)

        avg_accuracy = (qa_acc + math_acc + code_acc + summ_acc) / 4

        # Reasoning Quality and Clarity require human evaluation.
        # Using paper reference values as placeholders.
        # Replace these with actual human ratings from your evaluation.
        ref = PAPER_RESULTS[tech_label]
        reasoning = ref["reasoning"]
        clarity   = ref["clarity"]

        overall = composite_score(avg_accuracy, reasoning, clarity)

        experiment_results[tech_label] = {
            "accuracy":  round(avg_accuracy, 4),
            "reasoning": reasoning,
            "clarity":   clarity,
            "overall":   overall,
        }

        print(
            f"    Accuracy: {avg_accuracy*100:.1f}%  |  "
            f"Composite P: {overall*100:.1f}%\n"
        )

    # ── Print and save ────────────────────────────────────────────────────────
    print("=== Experiment Results ===")
    print_results_table(experiment_results)

    output_path = Path("results/experiment_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Results saved → {output_path}\n")


if __name__ == "__main__":
    main()
