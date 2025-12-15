from dataclasses import dataclass

from agents import RunContextWrapper, function_tool
from swerex.deployment.abstract import AbstractDeployment
from swerex.runtime.abstract import (
    BashAction,
    CreateBashSessionRequest,
    ReadFileRequest,
    ReadFileResponse,
    WriteFileRequest,
)

from adapter.problem import VerifiableProblem


@dataclass
class ProgrammingEnvironment[T: AbstractDeployment]:
    deployment: T

    @classmethod
    async def create(cls, deployment: T):
        env: ProgrammingEnvironment[T] = cls(deployment=deployment)
        await env.deployment.start()
        await env.deployment.runtime.create_session(CreateBashSessionRequest())
        return env

    async def is_passing(self, problem: VerifiableProblem) -> bool:
        """Verifies the provided problem by running the test code against the canonical solution."""
        runtime = self.deployment.runtime

        # Prepare the test code
        await runtime.write_file(
            WriteFileRequest(path="test.py", content=problem.test_code)
        )

        # Execute the test script
        await runtime.write_file(
            WriteFileRequest(path="test.py", content=problem.test_code)
        )
        result = await runtime.run_in_session(BashAction(command="python test.py"))
        success = result.exit_code == 0

        return success


@function_tool
async def read_file(
    wrapper: RunContextWrapper[ProgrammingEnvironment], path: str
) -> str:
    """Reads the content of a file at the given path within the programming environment."""
    runtime = wrapper.context.deployment.runtime

    result: ReadFileResponse = await runtime.read_file(
        request=ReadFileRequest(path=path)
    )
    return result.content


@function_tool
async def write_file(
    wrapper: RunContextWrapper[ProgrammingEnvironment], path: str, content: str
) -> None:
    """Writes the given content to a file at the specified path within the programming environment."""
    runtime = wrapper.context.deployment.runtime

    await runtime.write_file(request=WriteFileRequest(path=path, content=content))


@function_tool
async def run_command(
    wrapper: RunContextWrapper[ProgrammingEnvironment], bash_command: str
) -> str:
    """Runs a command in the programming environment and returns its output."""
    runtime = wrapper.context.deployment.runtime

    result = await runtime.run_in_session(BashAction(command=bash_command))
    return result.output


@function_tool
async def submit() -> str:
    """Finalizes the programming environment session."""
    return "Submission complete."
