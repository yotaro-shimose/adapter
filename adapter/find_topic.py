from adapter.models.topics import Topics
from oai_utils.mcp.filesystem import filesystem_mcp
from pathlib import Path
from textwrap import dedent

from agents.model_settings import ModelSettings
from oai_utils.agent import AgentWrapper


async def find_topics(local_path: Path, file_path: str) -> Topics:
    async with filesystem_mcp([str(local_path)]) as filesystem:
        agent = AgentWrapper[Topics].create(
            name="topic_finder",
            instructions=dedent(
                """\
You are a curriculum designer.
Given a local path of cloned repository and a specific file path inside the repository, your task is to create a curriculum for learners for the library based on the content of the provided file path.
You have access to a filesystem MCP server that allows you to query repository contents.
You should get the content of the specified file and respond with an exhaustive list of concepts or topics in the document to learn for the new library user.

Each topic will be used for creating exercises by another process.
Some document includes information not useful for library learnes such as contribution guides. You can return empty list in that case.

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
                filesystem,
            ],
            model="gpt-5-mini",
            output_type=Topics,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )
        result = await agent.run(
            input=f"GitHub URL: {local_path}\nFilePath: {file_path}",
            max_turns=20,
        )
        return result.final_output()
