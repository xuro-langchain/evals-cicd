"""
LangSmith evaluation test for CI/CD.

This test runs the multi-agent system against a LangSmith dataset and evaluates
the results using LLM-as-judge. Results are saved to a config file that the
report script uses to generate PR comments.

Usage:
    pytest -m evaluator
"""

import json
import pytest
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_async_llm_as_judge

from app.graph import run_graph

# Initialize LangSmith client
client = Client()

# Custom prompt for semantic correctness (less strict on wording)
# Uses OpenEvals placeholders: {inputs}, {outputs}, {reference_outputs}
SEMANTIC_CORRECTNESS_PROMPT = """You are evaluating whether an AI response contains the key facts from the expected answer.

<User Question>
{inputs}
</User Question>

<AI Response>
{outputs}
</AI Response>

<Expected Answer>
{reference_outputs}
</Expected Answer>

Evaluate based on SEMANTIC CORRECTNESS:
- Does the AI response contain the same KEY FACTS as the expected answer?
- Minor wording differences are OK (e.g., "AC/DC has albums X and Y" vs "Albums by AC/DC: X, Y")
- Additional helpful information beyond expected is OK
- The response should NOT contradict or omit key facts from expected

Score 1 if the response contains all key facts from expected (even with different phrasing).
Score 0 if key facts are missing, incorrect, or contradicted.

Respond with just the score (0 or 1) and a brief reason."""

# Create LLM-as-judge evaluator with lenient prompt
judge_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
correctness_evaluator = create_async_llm_as_judge(
    prompt=SEMANTIC_CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=judge_model,
)


@pytest.mark.evaluator
@pytest.mark.asyncio
async def test_multiagent_evaluation():
    """
    Run evaluation against a LangSmith dataset.

    Prerequisites:
    1. Create a dataset in LangSmith with question/answer pairs
    2. Set LANGSMITH_API_KEY environment variable
    3. Update DATASET_NAME below to match your dataset

    Dataset format:
    - Input: {"messages": [{"role": "user", "content": "your question"}]}
    - Output: {"output": "expected answer"}
    """
    # Configuration
    DATASET_NAME = "CICD Standalone: Multi-Agent Eval"
    EXPERIMENT_PREFIX = "multiagent-eval"
    PASSING_THRESHOLD = 0.7

    # Run evaluation
    experiment_results = await client.aevaluate(
        run_graph,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        num_repetitions=1,
        max_concurrency=5,
    )

    assert experiment_results is not None
    print(f"Evaluation completed: {experiment_results.experiment_name}")

    # Define scoring criteria
    criteria = {"correctness": f">={PASSING_THRESHOLD}"}

    # Save config for report generation
    output_metadata = {
        "experiment_name": experiment_results.experiment_name,
        "criteria": criteria,
    }

    safe_name = experiment_results.experiment_name.replace(":", "-").replace("/", "-")
    config_filename = f"evaluation_config__{safe_name}.json"

    with open(config_filename, "w") as f:
        json.dump(output_metadata, f, indent=2)

    print(f"Config saved: {config_filename}")
