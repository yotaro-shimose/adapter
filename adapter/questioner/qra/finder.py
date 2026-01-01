from oai_utils.agent import AgentsSDKModel
from oai_utils.client import get_aoai
from pathlib import Path
from textwrap import dedent

from agents.model_settings import ModelSettings
from oai_utils.agent import AgentWrapper
from oai_utils.mcp.filesystem import filesystem_mcp

from adapter.utils.savable import Savable


class FilePathsList(Savable):
    file_paths: list[str]


async def list_document_filepaths(
    local_dir: Path, model: AgentsSDKModel | None = None
) -> FilePathsList:
    if model is None:
        model = get_aoai("gpt-5-mini")
    async with filesystem_mcp([str(local_dir)]) as filesystem:
        agent = AgentWrapper[FilePathsList].create(
            name="file_path_finder",
            instructions=dedent(
                """\
You are a file path finder.
Given a local path of clone repository, your task is to list all the useful documentation files for library learners in the repository.
You have access to a file system mcp server that allows you to query repository contents.

The list should be exhaustive.
But you should not include documents which does not include useful information for library leaners such as contributing guide, code of conduct and changelogs.

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
                filesystem,
            ],
            model=model,
            output_type=FilePathsList,
            model_settings=ModelSettings(parallel_tool_calls=True),
        )
        result = await agent.run(
            input=f"Local path: {local_dir}",
            max_turns=30,
            time_out_seconds=600,
        )
        return result.final_output()
