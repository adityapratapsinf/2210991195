"""
tests/test_prompting_techniques.py
===================================
Tests the prompt-building logic without making any real API calls.
Uses unittest.mock to intercept the OpenAI call.

Run with:  python -m pytest tests/ -v

Author : Aditya Pratap Singh (Roll: 2210991195)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import prompting_techniques as pt


def _mock_call(prompt: str, temperature=0) -> str:
    """Return the prompt back so we can inspect what was built."""
    return f"MOCK_RESPONSE::{prompt}"


# Patch the actual OpenAI call in all tests
@patch.object(pt, "call_gpt4", side_effect=_mock_call)
class TestZeroShot:
    def test_passes_task_directly(self, mock_fn):
        result = pt.zero_shot("What is 2+2?")
        assert "What is 2+2?" in result


@patch.object(pt, "call_gpt4", side_effect=_mock_call)
class TestFewShot:
    def test_includes_examples_in_prompt(self, mock_fn):
        examples = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        result = pt.few_shot("Q3", examples)
        assert "Q1" in result
        assert "A1" in result
        assert "Q3" in result

    def test_output_label_appended(self, mock_fn):
        result = pt.few_shot("my question", [{"input": "x", "output": "y"}])
        assert "Output:" in result


@patch.object(pt, "call_gpt4", side_effect=_mock_call)
class TestChainOfThought:
    def test_zero_shot_cot_trigger(self, mock_fn):
        result = pt.chain_of_thought("What is 3+5?")
        assert "step by step" in result

    def test_few_shot_cot_includes_reasoning(self, mock_fn):
        examples = [{"input": "Q", "reasoning": "Because X", "output": "Y"}]
        result = pt.chain_of_thought("my question", few_shot_examples=examples)
        assert "Because X" in result
        assert "Reasoning:" in result


@patch.object(pt, "call_gpt4", side_effect=_mock_call)
class TestRoleBased:
    def test_persona_prepended(self, mock_fn):
        persona = "You are an expert physicist."
        result = pt.role_based("Explain gravity.", persona)
        assert persona in result
        assert "Explain gravity." in result


@patch.object(pt, "call_gpt4", side_effect=_mock_call)
class TestStructured:
    def test_all_components_present(self, mock_fn):
        result = pt.structured(
            task_steps=["Step one", "Step two"],
            persona="You are an expert.",
            constraints=["No jargon", "Be concise"],
            output_format="Bullet points",
        )
        assert "You are an expert." in result
        assert "Step one" in result
        assert "Step two" in result
        assert "No jargon" in result
        assert "Bullet points" in result

    def test_steps_are_numbered(self, mock_fn):
        result = pt.structured(
            task_steps=["Do A", "Do B"],
            persona="Expert",
            constraints=["Rule 1"],
            output_format="JSON",
        )
        assert "1." in result
        assert "2." in result
