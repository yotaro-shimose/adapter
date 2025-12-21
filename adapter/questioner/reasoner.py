from pydantic import BaseModel
from adapter.models.problems import QAProblem
from oai_utils.agent import AgentWrapper
from agents.mcp.server import MCPServerStdio
from agents.model_settings import ModelSettings
from pathlib import Path


class ReasoningOutput(BaseModel):
    reasoning: str


async def hindsight_reasoning(
    qa: QAProblem,
    local_dir: Path,
    file_path: str,
    filesystem_mcp: MCPServerStdio,
    max_turns: int = 20,
) -> str:
    prompt = """\
You are an expert Python developer with a deep, perfect memory of library APIs and internal mechanics.
Your task is to provide a "hindsight reasoning" process for a given question and its answer.

### Instructions:
1.  **Assume the Persona**: You are an expert who understands the library's design philosophy, performance trade-offs, and implementation details.
2.  **Hindsight Thinking**:
    - Start by looking at the question as if you are thinking from scratch.
    - However, because you know the 'answer', you must construct a reasoning path that logically and elegantly leads to that answer.
    - Your reasoning should feel like a "stream of consciousness" from an expert who is connecting the dots between the question's scenario and the library's underlying principles.
3.  **Depth**: Don't just restate the answer. Explain the "why" and "how". Connect the question to broader library concepts or common pitfalls.
4.  **Verification (Optional)**: Use the File System MCP ONLY if needed to examine the library's source code, base class definitions, or related utility functions to ensure your reasoning is technically accurate and well-supported. If the question and answer are already clear and you're confident in the mechanics, you don't have to use MCP tools.
5.  **Formatting**: Output the reasoning process in the `reasoning` field of the structured output. Do not include the original question or answer in your output, just the reasoning process itself."""

    agent = AgentWrapper[ReasoningOutput].create(
        name="hindsight_reasoner",
        instructions=prompt,
        model="gpt-5-mini",
        output_type=ReasoningOutput,
        mcp_servers=[filesystem_mcp],
        model_settings=ModelSettings(parallel_tool_calls=True),
    )

    input_text = f"""\
Local path: {local_dir}
File path: {file_path}
Question: {qa.question}
Answer: {qa.answer}"""

    ret = await agent.run(
        input=input_text,
        max_turns=max_turns,
    )
    return ret.final_output().reasoning
