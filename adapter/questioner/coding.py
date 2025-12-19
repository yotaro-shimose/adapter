from agents.model_settings import ModelSettings
from adapter.questioner.shared_prompts import DO_NOT_REFER_TO_DOCUMENT
from oai_utils.agent import AgentWrapper
from adapter.models.problems import VerifiableProblem
from pathlib import Path
from agents.mcp.server import MCPServerStdio
from adapter.models.topics import Topic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class ProblemVerificationError(BaseException):
    pass


@retry(
    retry=retry_if_exception_type(ProblemVerificationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def create_coding_task(
    local_dir: Path, file_path: str, topic: Topic, filesystem_mcp: MCPServerStdio
) -> VerifiableProblem:
    agent = AgentWrapper[VerifiableProblem].create(
        name="problem_generator",
        instructions=f"""\
You are an expert Python teacher.
Your goal is to create a coding assessment task based on the provided library documentation snippet and provided topic.
The task should test whether a developer understands how to use the specific functions, classes, or parameters described in the documentation.
{DO_NOT_REFER_TO_DOCUMENT}


### Output JSON Schema
You must output a single JSON object containing the task details.
{{
    "task_name": "Short descriptive name of the task",
    "problem_statement": "Clear instructions for the developer. Describe what the code should define. Mention specific parameters or behaviors found in the docs. It should notify and assume the user's functions and/or classes are available as `from submission import <symbol names>`.",
    "canonical_solution": "A complete, working Python solution (imports + function definition). This is the 'Gold' answer.",
    "test_code": "A standalone Python script to verify the solution. The code should import the user's submission and test it. We will treat it successful if this script runs without errors."
}}

### File System MCP
While the generated task should be based on the content of provided file path, you can refer to the other parts of the repository to get further context if needed (not always) by using file system functions.
""",
        model="gpt-5",
        output_type=VerifiableProblem,
        model_settings=ModelSettings(parallel_tool_calls=True),
        mcp_servers=[filesystem_mcp],
    )
    ret = await agent.run(
        input=f"""\
            Local path: {local_dir}
            File path: {file_path}
            Topic title: {topic.title}
            Topic description: {topic.description}
            """,
        max_turns=20,
    )

    problem = ret.final_output()
    # is_valid = await verify_problem(problem, deployment)
    # if not is_valid:
    #     raise ProblemVerificationError("Generated problem failed verification.")
    return problem
