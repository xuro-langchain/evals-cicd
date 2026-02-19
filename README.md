# LangSmith Evaluations in CI/CD

A minimal example showing how to run LangSmith evaluations for a **multi-agent system** as part of your CI/CD pipeline. When you open a PR, the workflow automatically:

1. Runs your multi-agent system against a test dataset
2. Evaluates responses using LLM-as-judge
3. Posts evaluation results as a PR comment

## Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      ┌──────────────┐          ┌─────────────┐
      │ Invoice Agent │          │ Music Agent │
      └──────────────┘          └─────────────┘
```

The supervisor routes customer queries to specialized agents:
- **Invoice Agent**: Handles billing, invoices, purchase history
- **Music Agent**: Handles music discovery, album/track searches

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── graph.py          # Multi-agent graph (supervisor + 2 subagents)
│   └── tools.py          # Tools for each subagent
├── tests/
│   └── test_eval.py      # Evaluation test
├── scripts/
│   └── report_eval.py    # Generates PR comment from results
├── .github/
│   └── workflows/
│       └── evaluate.yml  # GitHub Actions workflow
├── pyproject.toml
└── .env.example
```

## Setup

### 1. Create a LangSmith Dataset

Create a dataset in [LangSmith](https://smith.langchain.com) with test cases. Each example should have:

**Input:**
```json
{"messages": [{"role": "user", "content": "What albums does AC/DC have?"}]}
```

**Output:**
```json
{"output": "AC/DC has albums including Back in Black, Highway to Hell..."}
```

Update `DATASET_NAME` in `tests/test_eval.py` to match your dataset name.

### 2. Configure GitHub Secrets

Add these secrets to your repository (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `LANGSMITH_API_KEY` | Your LangSmith API key |

Create a GitHub environment called `production` and add the secrets there.

### 3. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 4. Run Locally

```bash
# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run evaluations
uv run pytest -m evaluator
```

## How It Works

### Evaluation Flow

```
PR Opened
    │
    ▼
┌─────────────────────────────────┐
│  evaluate job                   │
│  - Run pytest -m evaluator      │
│  - Multi-agent handles queries  │
│  - LLM-as-judge scores results  │
│  - Save config JSON             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  report job                     │
│  - Download config artifacts    │
│  - Query LangSmith for scores   │
│  - Generate markdown report     │
│  - Post as PR comment           │
└─────────────────────────────────┘
```

### Key Components

**`app/graph.py`** - Multi-agent system using LangGraph:
- Supervisor agent routes queries to specialists
- Invoice agent with billing tools
- Music agent with catalog search tools

**`app/tools.py`** - Database tools using Chinook sample database:
- Invoice lookups
- Album/track searches

**`tests/test_eval.py`** - Runs the multi-agent against a LangSmith dataset using `client.aevaluate()`. Uses [OpenEvals](https://github.com/langchain-ai/openevals) for LLM-as-judge evaluation.

**`scripts/report_eval.py`** - Queries LangSmith for experiment results and generates markdown.

## Customization

### Add New Agents

1. Create tools in `app/tools.py`
2. Add a new agent in `app/graph.py`:
```python
new_agent = create_react_agent(
    model,
    tools=new_tools,
    name="new_agent",
    prompt="Your agent prompt...",
    state_schema=State,
)
```
3. Add to the supervisor's agent list

### Change Evaluators

```python
from openevals.prompts import HELPFULNESS_PROMPT

helpfulness_evaluator = create_async_llm_as_judge(
    prompt=HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    judge=judge_model,
)
```

### Change Thresholds

Update the `criteria` dict in `test_eval.py`:

```python
criteria = {
    "correctness": ">=0.8",  # 80% threshold
}
```

## Resources

- [LangSmith Documentation](https://docs.smith.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenEvals](https://github.com/langchain-ai/openevals)
