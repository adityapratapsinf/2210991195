# Prompt Engineering Techniques for LLMs

**Author:** Aditya Pratap Singh — Roll No. 2210991195  
**Mentor:** Mr. Gurpreet Singh  
**Institution:** Department of Computer Science and Engineering, Chitkara University, Punjab  
**Course:** CO-OP Project at Industry — Module 2 (22CS421)

---

## About

This repository contains the complete implementation code for the research paper:

> **"A Comparative Analysis of Prompt Engineering Techniques for Improving Large Language Model Responses"**

The project evaluates five prompting strategies on GPT-4 across four benchmark task types using a unified composite scoring framework.

---

## Techniques

| # | Technique | Description |
|---|-----------|-------------|
| 1 | Zero-Shot | Task instruction only — no examples or persona |
| 2 | Few-Shot | 3–5 demonstration input-output pairs prepended |
| 3 | Chain-of-Thought | Step-by-step reasoning before the final answer |
| 4 | Role-Based | Domain-expert persona prepended to the task |
| 5 | Structured | Role + task steps + constraints + output format |

---

## Results (Table I from the paper)

| Technique | Accuracy | Reasoning | Clarity | Overall (P) |
|-----------|:---:|:---:|:---:|:---:|
| Zero-Shot | 68% | 64% | 70% | 67% |
| Few-Shot | 78% | 75% | 80% | 78% |
| Role-Based | 82% | 80% | 84% | 82% |
| Chain-of-Thought | 86% | 89% | 85% | 87% |
| **Structured** | **91%** | **90%** | **92%** | **91%** |

**Scoring formula: P = 0.40 × A + 0.35 × Q + 0.25 × V**

---

## Project Structure

```
prompt-engineering-project/
│
├── src/
│   ├── prompting_techniques.py   # All 5 prompting functions
│   ├── evaluation.py             # Exact Match, ROUGE-L, composite score, Cohen's Kappa
│   └── datasets.py               # Dataset loaders + built-in sample data
│
├── tests/
│   ├── test_evaluation.py        # Unit tests for all metrics
│   └── test_prompting_techniques.py  # Unit tests for prompt-building logic
│
├── data/                         # Put your JSONL dataset files here
├── results/                      # Experiment output saved here (JSON)
│
├── run_experiment.py             # Main experiment runner
├── requirements.txt
├── .env.example                  # Copy to .env and add your API key
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/prompt-engineering-project.git
cd prompt-engineering-project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
cp .env.example .env
# Now open .env and replace the placeholder with your actual key
```

Your `.env` file should look like this:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> The `.env` file is listed in `.gitignore` — it will **never** be pushed to GitHub.

---

## Running the Experiments

**Using built-in sample data (no extra files needed):**
```bash
python run_experiment.py
```

**Using your own dataset files:**
```bash
python run_experiment.py \
    --qa   data/qa.jsonl \
    --math data/math.jsonl \
    --code data/code.jsonl \
    --summ data/summaries.jsonl
```

**Print paper reference results without any API calls:**
```bash
python run_experiment.py --paper-results
```

---

## Running the Tests

All tests run completely offline — no API key needed.

```bash
python -m pytest tests/ -v
```

Expected output: **32 passed**

---

## Dataset Format

Each file should be a JSONL file (one JSON object per line).

**Q&A**
```json
{"question": "What is the capital of France?", "answer": "Paris"}
```

**Maths**
```json
{"question": "Tom has 5 apples...", "answer": "8", "reasoning": "5 + 3 = 8"}
```

**Code**
```json
{"prompt": "def add(a, b):\n    ...\n", "canonical_solution": "...", "test": "assert add(1,2)==3"}
```

**Summarisation**
```json
{"article": "Scientists discovered...", "reference_summary": "Researchers found..."}
```

---

## Scoring Formula

From Section III.G of the paper:

```
P = 0.40 × A  +  0.35 × Q  +  0.25 × V
```

| Symbol | Meaning | Source |
|--------|---------|--------|
| A | Accuracy (Exact Match or ROUGE-L F1) | Automated |
| Q | Reasoning Quality (0–1) | Human evaluation |
| V | Response Clarity (0–1) | Human evaluation |

---

## References

1. T. B. Brown et al., "Language models are few-shot learners," NeurIPS, 2020.
2. J. Wei et al., "Chain-of-thought prompting elicits reasoning in LLMs," NeurIPS, 2022.
3. T. Kojima et al., "Large language models are zero-shot reasoners," NeurIPS, 2022.
4. P. Liu et al., "Pre-train, prompt, and predict," ACM Comput. Surv., 2023.
5. Z. Zhao et al., "Calibrate before use," ICML, 2021.
6. S. Min et al., "Rethinking the role of demonstrations," EMNLP, 2022.
