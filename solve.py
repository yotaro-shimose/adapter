from adapter.solver.constant import SOLVER_PROMPT
from oai_utils.tracing import setup_openai_tracing
import asyncio
from pathlib import Path
from oai_utils.vllm import VLLMSetup
from oai_utils.agent import AgentWrapper
from oai_utils.runresult import SimpleReasoningItem
from adapter.models.problems import QRADataset, QAProblem, QRA
from adapter.questioner.evaluater import evaluate_qa
from adapter.utils.savable import Savable
from async_utils import gather_with_semaphore


class SolveResult(Savable):
    qra: QRA
    agent_reasoning: str
    agent_answer: str
    is_correct: bool


class EvaluationResults(Savable):
    results: list[SolveResult]


async def solve_and_evaluate(agent: AgentWrapper[str], qra: QRA) -> SolveResult | None:
    try:
        # Run agent to get answer
        result = await agent.run(input=qra.question)
        agent_answer = result.final_output()
        simplified_items = result.output_with_reasoning().simplified()

        # Extract reasoning reasoning item (expecting at most 1)
        reasoning_items = [
            item.content
            for item in simplified_items
            if isinstance(item, SimpleReasoningItem)
        ]
        agent_reasoning = (
            reasoning_items[0] if reasoning_items else ""
        )  # User mentioned "at most 1"

        # Evaluate answer
        qa_problem = QAProblem(question=qra.question, answer=qra.answer)
        eval_result = await evaluate_qa(qa_problem, agent_answer, model="gpt-4o")

        return SolveResult(
            qra=qra,
            agent_reasoning=agent_reasoning,
            agent_answer=agent_answer,
            is_correct=eval_result.is_correct,
        )
    except Exception as e:
        # User requested to ignore tracing errors, but we still catch general errors
        # If it's a tracing error, it might still bubble up or look like noise
        print(f"Error solving and evaluating problem: {e}")
        return None


async def main():
    model_type = "Qwen"
    # model_type = "gpt-4o"
    # model_type = "Qwen-Finetuned"
    # model_type = "Qwen-Finetuned-Conv"
    data_parallel_size = 2
    model_type = "Qwen-Finetuned-Conv-bugfixed"

    setup_openai_tracing()
    reasoning_parser = "deepseek_r1"
    if model_type == "Qwen":
        model = VLLMSetup.qwen3(data_parallel_size=data_parallel_size)
        await model.ensure_vllm_running()
    elif model_type == "Qwen-Finetuned":
        model = VLLMSetup(
            model="checkpoints/merged/qwen3-4b-sqlglot-merged",
            reasoning_parser=reasoning_parser,
            data_parallel_size=data_parallel_size,
        )
        await model.ensure_vllm_running()
    elif model_type == "Qwen-Finetuned-Conv":
        model = VLLMSetup(
            model="checkpoints/merged/qwen3-4b-sqlglot-merged-conversational",
            reasoning_parser=reasoning_parser,
            data_parallel_size=data_parallel_size,
        )
        await model.ensure_vllm_running()
    elif model_type == "gpt-4o":
        model = "gpt-4o"
    elif model_type == "Qwen-Finetuned-Conv-bugfixed":
        model = VLLMSetup(
            model="Qwen/Qwen3-4B",
            reasoning_parser=reasoning_parser,
            lora_adapter="checkpoints/qwen3-4b-sqlglot-finetuned-2025-12-24",
            data_parallel_size=data_parallel_size,
        )
        await model.ensure_vllm_running()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load dataset
    dataset_path = Path("data/sqlglot/problems.json")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    qra_dataset = QRADataset.load(dataset_path).sort().head(12)
    print(f"Loaded {len(qra_dataset.problems)} problems from {dataset_path}")

    # Create agent for solving
    agent = AgentWrapper[str].create(
        name="sqlglot_solver",
        instructions=SOLVER_PROMPT,
        model=model,
    )

    # Evaluate all problems in parallel
    print(f"Starting parallel evaluation of {len(qra_dataset.problems)} problems...")

    tasks = [solve_and_evaluate(agent, qra) for qra in qra_dataset.problems]
    # Increase concurrency since we have data_parallel_size = 4
    results: list[SolveResult | None] = await gather_with_semaphore(
        tasks, max_concurrent=32, progressbar=True
    )

    valid_results = [res for res in results if res is not None]
    correct_count = sum(1 for res in valid_results if res.is_correct)
    total_count = len(valid_results)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\nEvaluation Complete!")
    print(f"Total Problems (Processed): {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save results to JSON
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"evaluation_results_{model_type}.json"

    print(f"Saving results to {output_file}...")
    evaluation_results = EvaluationResults(results=valid_results)
    evaluation_results.save(output_file)


if __name__ == "__main__":
    asyncio.run(main())
