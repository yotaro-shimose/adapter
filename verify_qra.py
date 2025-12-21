from adapter.verifier.verify import verify_qra
from adapter.utils.async_util import gather_with_semaphore
from pathlib import Path
from adapter.models.problems import QRADataset


async def main():
    problems = Path("./sqlglot_problems.json")
    if not problems.exists():
        print(f"File {problems} not found.")
        return

    problems = QRADataset.load(problems)
    print(f"Found {len(problems.problems)} problems. Starting verification...")

    verifications = await gather_with_semaphore(
        [verify_qra(problem) for problem in problems.problems], 10
    )

    for problem, (is_valid, feedback) in zip(problems.problems, verifications):
        status = "✅ [VALID]" if is_valid else "❌ [INVALID]"
        print("-" * 80)
        print(f"Question: {problem.question}")
        print(f"Reasoning: {problem.reasoning}")
        print(f"Answer: {problem.answer}")
        print(f"Status: {status}")
        print(f"Feedback: {feedback}")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
