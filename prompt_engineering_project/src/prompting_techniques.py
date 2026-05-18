"""
prompting_techniques.py
=======================
Implements the five prompting strategies evaluated in the paper:

    1. Zero-Shot       — task only, no examples
    2. Few-Shot        — k demonstration pairs before the task
    3. Chain-of-Thought — step-by-step reasoning before the answer
    4. Role-Based      — domain-expert persona prepended to the task
    5. Structured      — role + task steps + constraints + output format

Reference:
    "A Comparative Analysis of Prompt Engineering Techniques for
     Improving Large Language Model Responses"

Author  : Aditya Pratap Singh  (Roll: 2210991195)
Mentor  : Mr. Gurpreet Singh
Org     : Chitkara University, Punjab
"""

import os
from openai import OpenAI


def _get_client() -> OpenAI:
    """Return an OpenAI client using the key from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Paste your key into .env\n"
            "  3. Re-run the script."
        )
    return OpenAI(api_key=api_key)


def call_gpt4(prompt: str, temperature: float = 0) -> str:
    """
    Send a single prompt to GPT-4 and return the text response.

    Args:
        prompt      : the full prompt string to send
        temperature : 0 = deterministic (used in all paper experiments)

    Returns:
        Model response as a plain string.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Zero-Shot Prompting
# ─────────────────────────────────────────────────────────────────────────────

def zero_shot(task: str) -> str:
    """
    The simplest baseline — just send the task to the model with no
    examples, no persona, and no extra guidance.

    Formal definition (from the paper):
        y = M(p),  where p = T  (task description only)

    Token cost: ~35–80 input tokens.

    Args:
        task : plain task description / question

    Returns:
        Model response string.
    """
    return call_gpt4(task)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Few-Shot Prompting
# ─────────────────────────────────────────────────────────────────────────────

def few_shot(task: str, examples: list[dict]) -> str:
    """
    Prepend k demonstration input-output pairs before the target query.
    The examples calibrate output format, length, and structure.

    Formal definition (from the paper):
        p = {(x1, y1), ..., (xk, yk), x_new}
        Typical k: 3–5

    Args:
        task     : the actual question / task
        examples : list of dicts, each with keys 'input' and 'output'

    Returns:
        Model response string.
    """
    demo_block = ""
    for ex in examples:
        demo_block += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt = f"{demo_block}Input: {task}\nOutput:"
    return call_gpt4(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chain-of-Thought (CoT) Prompting
# ─────────────────────────────────────────────────────────────────────────────

def chain_of_thought(
    task: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """
    Ask the model to show its reasoning step by step before giving
    the final answer. This concentrates probability mass on correct
    reasoning chains.

    Supports two modes:
      - Zero-Shot CoT : appends 'Let us think step by step.' (Kojima et al. 2022)
      - Few-Shot CoT  : includes worked reasoning traces as examples

    Formal definition (from the paper):
        p = (T, r1, r2, ..., rm, a)
        P(a* | p) = sum_R P(a* | R, p) * P(R | p)

    Args:
        task              : the task or question
        few_shot_examples : optional list of dicts with keys
                            'input', 'reasoning', 'output'

    Returns:
        Model response string.
    """
    if few_shot_examples:
        demo_block = ""
        for ex in few_shot_examples:
            demo_block += (
                f"Question: {ex['input']}\n"
                f"Reasoning: {ex['reasoning']}\n"
                f"Answer: {ex['output']}\n\n"
            )
        prompt = f"{demo_block}Question: {task}\nReasoning:"
    else:
        prompt = f"{task}\n\nLet us think step by step."

    return call_gpt4(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Role-Based Prompting
# ─────────────────────────────────────────────────────────────────────────────

def role_based(task: str, persona: str) -> str:
    """
    Prepend a domain-expert persona to the task description.
    This steers the model's output distribution toward expert-quality text.

    Formal definition (from the paper):
        D_0 | role(e) ≈ D_r
        where D_r = expert text distribution for domain d

    Token cost: ~120–280 input tokens.

    Args:
        task    : the task or question
        persona : expert role string, e.g.
                  'You are a senior data scientist with 10 years of
                   experience in machine learning and statistical inference.'

    Returns:
        Model response string.
    """
    prompt = f"{persona}\n\n{task}"
    return call_gpt4(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Structured Prompting
# ─────────────────────────────────────────────────────────────────────────────

def structured(
    task_steps: list[str],
    persona: str,
    constraints: list[str],
    output_format: str,
) -> str:
    """
    Build a structured prompt from four explicit components:
        R — Role assignment (persona)
        T — Task decomposition (numbered sub-steps)
        C — Constraints (hard rules the model must follow)
        F — Output format specification

    This is the highest-performing technique in the paper (P = 0.909).
    It systematically reduces output entropy by constraining all four
    major sources of output variance.

    Formal definition (from the paper):
        p_struct = (R, T, C, F)

    Token cost: ~250–450 input tokens.

    Args:
        task_steps    : ordered list of sub-tasks / steps
        persona       : expert role string
        constraints   : list of hard rules / restrictions
        output_format : description of the exact expected output structure

    Returns:
        Model response string.
    """
    steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(task_steps))
    constraints_block = "\n".join(f"  - {c}" for c in constraints)

    prompt = (
        f"{persona}\n\n"
        f"Please complete the following tasks:\n{steps_block}\n\n"
        f"Constraints:\n{constraints_block}\n\n"
        f"Output format:\n  {output_format}"
    )
    return call_gpt4(prompt)
