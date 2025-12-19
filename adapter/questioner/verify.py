from swerex.runtime.abstract import Command
from swerex.runtime.abstract import WriteFileRequest
from swerex.runtime.abstract import CreateBashSessionRequest
from adapter.models.problems import VerifiableProblem
from swerex.deployment.abstract import AbstractDeployment


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
