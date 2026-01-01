from pydantic import BaseModel
from oai_utils.agent import AgentWrapper, AgentRunFailure, AgentsSDKModel
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type
from adapter.models.problems import QAProblem
from agents.mcp.server import MCPServerStdio
from adapter.topic.topics import Topic
from pathlib import Path
from agents.model_settings import ModelSettings


async def create_qa(
    local_dir: Path,
    file_path: str,
    topic: Topic,
    filesystem_mcp: MCPServerStdio,
    model: AgentsSDKModel,
    max_turns: int = 20,
) -> QAProblem | None:
    prompt = f"""\
You are an expert Python technical interviewer and library documentation specialist.
Your goal is to create a high-quality, self-contained question and answer pair based on the provided library documentation snippet.

### Guidelines for a "Self-Contained" Question:
1.  **Contextual Setup**: Provide enough context to establish the scenario (e.g., "A developer is using the pandas Dataframe API to merge datasets", "When finetuning Llama with Huggingface Transformers"). You must identify the specific classes, methods, modules or concepts involved so the question is self-contained.
    HOWEVER, you must not explain the logic or rules of those methods (e.g., do not explain the difference between an inner or outer join, or how a specific aggregation is calculated).
    The solver should deduce the behavior based on their knowledge of the library's design.
2.  **Conceptual over Verbatim**: Test the usage, concepts of the library instead of asking specific document content verbatim. You should not mention `The README` or `The documentation`.
3.  **No Meta-References**: Do not use phrases like "According to the document" or "As mentioned in the snippet." as solver has no access to the document and is not assumed to remember in verbatim manner.

### Output JSON Schema
You must output a single JSON object:
{
        "question": "A comprehensive, self-contained scenario-based question including necessary context and code snippets.",
    "answer": "A detailed explanation that clarifies the mechanics and consequences of the library's behavior."
}

### Research Task
Base the task on the content of the provided file path. Use the File System MCP to examine the base class definitions or related utility functions to ensure the 'answer' is technically accurate.
You have {max_turns} turns to complete the task."""
    agent = AgentWrapper[QAProblem].create(
        name="qa_generator",
        instructions=prompt,
        model=model,
        output_type=QAProblem,
        mcp_servers=[filesystem_mcp],
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    input = f"""\
Local path: {local_dir}
File path: {file_path}
Topic title: {topic.title}
Topic description: {topic.description}"""
    ret = await agent.run(
        input=input,
        max_turns=max_turns,
    )
    return ret.final_output()


class QAGenerationOutput(BaseModel):
    tasks: list[QAProblem]


async def create_multiple_qas(
    local_dir: Path,
    file_path: str,
    topic: Topic,
    filesystem_mcp: MCPServerStdio,
    model: AgentsSDKModel,
    max_turns: int = 20,
) -> list[QAProblem]:
    prompt = """\
You are an expert Python technical interviewer and library documentation specialist.
Your goal is to create one or more high-quality, self-contained question and answer pairs based on the provided library documentation snippet.

### Core Objectives:
- **Atomic Scope**: Each Q&A pair must focus on a single specific architectural concept or mechanic. If a document is complex, generate multiple separate Q&A pairs.
- **Identify, Don't Explain**: Provide enough context to establish the scenario, but do not give away the logic, rules, or definitions. The solver must provide the expertise.
- **Test for Recall**: Design questions that require the solver to provide specific syntax, command names, or variable names from memory based on a described goal.

### Guidelines for "Self-Contained" Questions:
1. **Contextual Setup**: Provide enough architectural context to establish the scenario (e.g., "A developer is implementing a SQL-to-SQL transpilation pipeline that must handle cross-dialect identifier quoting"; or "A developer is writing a custom optimization pass to identify and remove redundant logical conditions in a SELECT statement"; or "A developer is programmatically building a complex query and needs to ensure that specific metadata is preserved during node replacement"). You must identify the specific modules or concepts involved so the question is self-contained. HOWEVER, you must not explain the logic or rules (e.g., do not explain how the quote character is chosen or how the replacement logic maintains parent pointers).
2. **Goal-Oriented Scenarios**: Instead of asking "What are the targets?", frame the question as a specific task. Describe the desired outcome (e.g., "The developer wants to run only core logic tests while ensuring a specific dependency is bypassed").
3. **No Meta-References**: Do not use phrases like "According to the document" as the solver has no access to it.

### Examples of Desired Granularity and Depth:

**Example 1 (Focus: Selective Testing & State)**
- **Question**: "A developer wants to run the project's unit tests. However, they do not have the Rust development environment installed and need to ensure the test runner specifically skips any tests that require the native extension. What is the exact command and environment variable pair used to achieve this 'Python-only' test run?"
- **Answer**: "The developer should run `SQLGLOTRS_TOKENIZER=0 make unit`. Setting the environment variable to 0 forces the pure-Python implementation, and the 'unit' target runs the standard library test suite."

**Example 2 (Focus: Dependency Management)**
- **Question**: "The project supports an alternative, faster Python package installer via a specific Makefile toggle. How does a contributor invoke the installation target so that it uses this optimized wrapper instead of standard pip?"
- **Answer**: "The contributor should prefix the command with the `UV=1` environment variable, for example: `UV=1 make install-dev`. This tells the Makefile to alias the PIP command to 'uv pip'."

**Example 3 (Focus: Structural Metadata)**
- **Question**: "When inspecting an AST node's representation, some optional child keys are hidden to save space. Which specific instance method must be called to produce a fully verbose textual output that reveals every argument, including empty keys and object IDs?"
- **Answer**: "The developer should call the `.to_s(verbose=True)` method. While `repr()` provides a compact view, `to_s` is the internal printer that supports a 'verbose' flag for debugging."

### Output JSON Schema
You must output a single JSON object containing a list of tasks:
{
    "tasks": [
        {
            "question": "A focused, self-contained scenario-based question that describes a goal without naming the solution.",
            "answer": "A detailed explanation providing the specific syntax, targets, or variables required."
        }
    ]
}

### Research Task
Base the tasks on the provided file path. Use the File System MCP to verify the exact names of targets, variables, or private methods to ensure the 'answer' is technically accurate.
"""
    agent = AgentWrapper[QAGenerationOutput].create(
        name="qa_generator",
        instructions=prompt,
        model=model,
        output_type=QAGenerationOutput,
        mcp_servers=[filesystem_mcp],
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    input = f"""\
Local path: {local_dir}
File path: {file_path}
Topic title: {topic.title}
Topic description: {topic.description}"""
    ret = await agent.run(
        input=input,
        max_turns=max_turns,
    )
    return ret.final_output().tasks


async def create_multiple_qas_retriable(
    local_dir: Path,
    file_path: str,
    topic: Topic,
    filesystem_mcp: MCPServerStdio,
    model: AgentsSDKModel,
    max_turns: int = 20,
) -> list[QAProblem]:
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(AgentRunFailure),
            reraise=True,
        ):
            with attempt:
                return await create_multiple_qas(
                    local_dir, file_path, topic, filesystem_mcp, model, max_turns
                )
    except AgentRunFailure:
        return []
    return []  # Should not reach here
