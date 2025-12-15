from dataclasses import dataclass
from textwrap import dedent
from typing import Self

from agents.agent import StopAtTools
from oai_wrapper.agent import AgentWrapper
from swerex.deployment.abstract import AbstractDeployment

from adapter.env import (
    ProgrammingEnvironment,
    read_file,
    run_command,
    submit,
    write_file,
)
from adapter.problem import VerifiableProblem


@dataclass
class ProblemSolver:
    agent: AgentWrapper[str]

    @classmethod
    def create(cls) -> Self:
        agent = AgentWrapper[str].create(
            name="programming_environment_agent",
            instructions=dedent("""\
        You are a programming environment agent that can read and write files, and run commands
        within a given programming environment.
        You have access to the following tools:
        1. read_file(path: str) -> str: Reads the content of a file at the given path.
        2. write_file(path: str, content: str) -> None: Writes the given content to a file at the specified path.
        3. run_command(bash_command: str) -> str: Runs a command in the environment and returns its output.
        4. submit() -> str: Finalizes the programming environment session. Should be called when you are done.
        """),
            model="gpt-5-mini",
            tool_use_behavior=StopAtTools(stop_at_tool_names=["submit"]),
            tools=[
                read_file,
                write_file,
                run_command,
                submit,
            ],
        )
        return cls(agent=agent)

    async def solve[T: AbstractDeployment](
        self, problem_statement: VerifiableProblem, env: ProgrammingEnvironment[T]
    ) -> ProgrammingEnvironment[T]:
        await self.agent.run(
            input=f"""\
            Here is the problem statement: {problem_statement.as_md()}
        """,
            context=env,
        )
        return env
