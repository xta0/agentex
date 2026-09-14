import asyncio
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel
from litellm import acompletion

load_dotenv()

class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

gaia_prompt = """
You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and
set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in
"final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in
"unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma
separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither
use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for
cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending on
whether the element is a number or a string.
"""

PROVIDER_SEMAPHORES = {
    "openai": asyncio.Semaphore(30),
    "anthropic": asyncio.Semaphore(10),
}

def get_provider(model: str) -> str:
    return "anthropic" if model.startswith("anthropic/") else "openai"

async def solve_problem(model: str, question: str) -> GaiaOutput:
    provider = get_provider(model)
    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": gaia_prompt},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason="The model refused to answer the question : " + finish_reason,
            )
        return GaiaOutput.model_validate_json(content)

def is_correct(prediction: str | None, answer: str) -> bool:
    if prediction is None:
        return False
    # Normalize the prediction and answer for comparison
    normalized_prediction = prediction.strip().lower()
    normalized_answer = answer.strip().lower()
    return normalized_prediction == normalized_answer

async def evaluatate_gaia_single(problem: dict, model: str) -> dict:
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "predicted_answer": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": False,
            "predicted_answer": None,
            "answer": problem["Final answer"],
            "unsolvable_reason": str(e),
        }

async def run_experiment(problems: list[dict], models: list[str]) -> dict[str, list]:
    tasks = [
        evaluatate_gaia_single(problem, model)
        for problem in problems
        for model in models
    ]
    all_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating GAIA problems")

    results = {models: [] for models in models}
    for result in all_results:
        results[result["model"]].append(result)
    return results
        

async def main():
    level1_problems = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_level1",
        split="validation",
    )
    print(f"Number of Level 1 problems: {len(level1_problems)}")
    MODELS = [
        "gpt-5",
        "gpt-5-mini",
    ]
    subset = level1_problems.select(range(20))
    results = await run_experiment(subset, MODELS)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
