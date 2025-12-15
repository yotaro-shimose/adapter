import os
from textwrap import dedent

import httpx
from agents.mcp.server import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from agents.model_settings import ModelSettings
from oai_wrapper.agent import AgentWrapper
from pydantic import BaseModel
from swerex.deployment.abstract import AbstractDeployment
from swerex.runtime.abstract import (
    Command,
    CreateBashSessionRequest,
    WriteFileRequest,
)

from adapter.problem import VerifiableProblem
from adapter.utils.savable import Savable


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


async def create_question(document: str) -> VerifiableProblem:
    questioner = AgentWrapper[VerifiableProblem].create(
        name="problem_generator",
        instructions=dedent("""\
You are an expert Python technical interviewer and library documentation specialist.
Your goal is to create a coding assessment task based *strictly* on the provided library documentation snippet.

You must output a single JSON object containing the task details. The task should test whether a developer understands how to use the specific functions, classes, or parameters described in the documentation.

### Output JSON Schema
{
    "task_name": "Short descriptive name of the task",
    "function_name": "Name of the function the user must implement (e.g., 'process_data')",
    "problem_statement": "Clear instructions for the developer. Describe what the function should take as input and what it should return. Mention specific parameters or behaviors found in the docs. It should notify and assume the user's function is available as `from submission import {function_name}`.
    "canonical_solution": "A complete, working Python solution (imports + function definition). This is the 'Gold' answer.",
    "test_code": "A standalone Python script to verify the solution. It must use assertions or `unittest` to verify correct behavior."
}

### Guidelines
1. **Relevance**: The task must focus on the specific features mentioned in the provided text. Do not ask for generic Python tasks.
2. **Self-Contained**: The `canonical_solution` and `test_code` must be valid and runnable. If dummy data is needed, generate it within the code.
3. **Verification**: The `test_code` must check the return values or side effects (e.g., file creation) strictly.
"""),
        model="gpt-5-mini",
        output_type=VerifiableProblem,
    )
    ret = await questioner.run(
        input=dedent(
            f"""
Based on the following library documentation, generate a coding task and its verification code.

<document>
{document}
</document>
""",
        )
    )
    return ret.final_output()


def github_mcp() -> MCPServerStreamableHttp:
    github_pat = os.environ["GITHUB_PAT"]
    return MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_pat}"},
        )
    )


class Topic(BaseModel):
    title: str
    description: str


class Topics(Savable):
    topics: list[Topic]


async def find_topics(repo_url: httpx.URL, file_path: str) -> Topics:
    async with github_mcp() as github:
        agent = AgentWrapper[Topics].create(
            name="topic_finder",
            instructions=dedent(
                """\
You are a curriculum designer.
Given a GitHub repository URL and a specific file path, your task is to create a curriculum for learners for the library based on the content of the provided file path.
You have access to a GitHub MCP server that allows you to query repository contents.
You should first get the content of the specified file and respond with an exhaustive list of concepts or topics in the document to learn for the new library user.

Your response should be structured as following:
```json
[
    {
        "title": "Topic title",
        "description": "Detailed description of what the user is expected to learn"
    }
]
```

Based on your list, another agent will dive deeper for each topic and create exercises.
""",
            ),
            mcp_servers=[
                github,
            ],
            model="gpt-5-mini",
            output_type=Topics,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )
        result = await agent.run(
            input=f"GitHub URL: {repo_url}¥nFilePath: {file_path}",
            max_turns=20,
        )
        return result.final_output()


async def find_document(repo_url: httpx.URL) -> str:
    async with github_mcp() as github:
        agent = AgentWrapper[str].create(
            name="documentation_finder",
            instructions=dedent(
                """\
You are a GitHub documentation finder agent.
Given a GitHub repository URL, your task is to describe the structure of its documentation.
You have access to a GitHub MCP server that allows you to query repository contents.

After that, propose the curriculum path (e.g., README.md, docs/index.md) that would be most useful for learning about the library.
""",
            ),
            mcp_servers=[
                github,
            ],
            model="gpt-5-mini",
            output_type=str,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )
        result = await agent.run(
            input=f"GitHub URL: {repo_url}",
            max_turns=20,
        )

        return result.final_output()


async def questioner(
    repo_url: httpx.URL, file_path: str, topic: Topic
) -> VerifiableProblem:
    async with github_mcp() as github:
        agent = AgentWrapper[VerifiableProblem].create(
            name="problem_generator",
            instructions=dedent("""\
    You are an expert Python teacher.
    Your goal is to create a coding assessment task based *strictly* on the provided library documentation snippet and provided topic.

    You must output a single JSON object containing the task details. The task should test whether a developer understands how to use the specific functions, classes, or parameters described in the documentation.

    ### Output JSON Schema
    {
        "task_name": "Short descriptive name of the task",
        "problem_statement": "Clear instructions for the developer. Describe what the code should define. Mention specific parameters or behaviors found in the docs. It should notify and assume the user's functions and/or classes are available as `from submission import <symbol names>`.",
        "canonical_solution": "A complete, working Python solution (imports + function definition). This is the 'Gold' answer.",
        "test_code": "A standalone Python script to verify the solution. The code should import the user's submission and test it. We will treat it successful if this script runs without errors."
    }

    ### Guidelines
    1. **Relevance**: The task must focus on the specific features mentioned in the provided text. Do not ask for generic Python tasks.
    2. **Self-Contained**: The `canonical_solution` and `test_code` must be valid and runnable. If dummy data is needed, generate it within the code.
    3. **Verification**: The `test_code` must check the return values or side effects (e.g., file creation) strictly.
    4. **Github MCP**: While the generated task should test the understanding of the topics in the documentation, you can refer to the other parts of the repository to get further context if needed (not always).

    """),
            model="gpt-5-mini",
            output_type=VerifiableProblem,
            model_settings=ModelSettings(parallel_tool_calls=True),
            mcp_servers=[github],
        )
        ret = await agent.run(
            input=f"""\
            GitHub URL: {repo_url}
            File path: {file_path}
            Topic title: {topic.title}
            Topic description: {topic.description}
            """,
            max_turns=20,
        )

    return ret.final_output()


class FilePathsList(Savable):
    file_paths: list[str]


async def list_document_filepaths(repo_url: httpx.URL) -> FilePathsList:
    async with github_mcp() as github:
        agent = AgentWrapper[FilePathsList].create(
            name="file_path_finder",
            instructions=dedent(
                """\
You are a file path finder.
Given a GitHub repository URL, your task is to list all the useful documentation files for library learners in the repository.
You have access to a GitHub MCP server that allows you to query repository contents.

The list should be exhaustive. But you should not include documents which does not include useful information for library leaners such as contributing guide, code of conduct and changelogs.

Your response should be structured as following:
```json
{
    "file_paths": [
        "docs/index.md",
        "README.md",
        "docs/api.md"
    ]
}
```
""",
            ),
            mcp_servers=[
                github,
            ],
            model="gpt-5-mini",
            output_type=FilePathsList,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )
        result = await agent.run(
            input=f"GitHub URL: {repo_url}",
            max_turns=30,
            time_out_seconds=600,
        )
        return result.final_output()
