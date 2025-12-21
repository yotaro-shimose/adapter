from adapter.models.problems import QRA
from swerex.runtime.abstract import Command
from swerex.runtime.abstract import WriteFileRequest
from swerex.runtime.abstract import CreateBashSessionRequest
from adapter.models.problems import VerifiableProblem
from swerex.deployment.abstract import AbstractDeployment
from pydantic import BaseModel
from oai_utils.agent import AgentWrapper


async def verify_problem(
    problem: VerifiableProblem, deployment: AbstractDeployment
) -> bool:
    """Verifies the provided problem by running the test code against the canonical solution."""
    await deployment.start()
    runtime = deployment.runtime

    await runtime.create_session(CreateBashSessionRequest())
    # Prepare the submission code
    submission_code = problem.canonical_solution + "\n"

    await runtime.write_file(
        WriteFileRequest(path="submission.py", content=submission_code)
    )
    # Prepare the test code
    await runtime.write_file(
        WriteFileRequest(path="test.py", content=problem.test_code)
    )

    # Execute the test script
    result = await runtime.execute(Command(command=["python", "test.py"]))
    success = result.exit_code == 0
    await deployment.stop()

    return success


class QRAVerificationOutput(BaseModel):
    is_valid: bool
    feedback: str


async def verify_qra(qra: QRA) -> tuple[bool, str]:
    """Verifies the provided QRA triplet using an LLM to check for logical consistency of the reasoning."""
    prompt = """\
You are an expert technical evaluator. Your task is to verify the logical consistency of the "Reasoning" process in a Question-Reasoning-Answer (QRA) triplet.

### Context:
- The **Question** and **Answer** are already verified as technically correct and accurate.
- Your primary focus is on the **Reasoning** process.

### Evaluation Criteria for Reasoning:
1. **Logical Path**: Does the reasoning provide a clear, step-by-step path that starts from the question and logically arrives at the provided answer?
2. **Depth and Insight**: Does the reasoning explain the "why" and "how" behind the library's behavior, rather than just restating the answer?
3. **Internal Mechanics**: Does it correctly reference the relevant library design principles, API contracts, or internal mechanics that lead to the behavior?
4. **Expert Persona**: Does it feel like an expert's "stream of consciousness" connecting the dots?

### Output:
You must output a JSON object with:
- `is_valid`: A boolean indicating if the reasoning is logically sound and effectively bridges the question to the answer.
- `feedback`: A brief explanation of your decision, especially if the reasoning is weak, disconnected, or circular.
"""

    agent = AgentWrapper[QRAVerificationOutput].create(
        name="qra_evaluator",
        instructions=prompt,
        model="gpt-5-mini",
        output_type=QRAVerificationOutput,
    )

    input_text = f"""\
Question: {qra.question}
Reasoning: {qra.reasoning}
Answer: {qra.answer}
"""

    ret = await agent.run(input=input_text)
    output = ret.final_output()
    return output.is_valid, output.feedback
