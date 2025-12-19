from pydantic import BaseModel
from oai_utils.agent import AgentWrapper
from adapter.models.problems import QAProblem
from agents.mcp.server import MCPServerStdio
from adapter.models.topics import Topic
from pathlib import Path
from agents.model_settings import ModelSettings


async def create_qa(
    local_dir: Path, file_path: str, topic: Topic, filesystem_mcp: MCPServerStdio
) -> QAProblem | None:
    prompt = """\
You are an expert Python technical interviewer and library documentation specialist.
Your goal is to create a high-quality, self-contained question and answer pair based on the provided library documentation snippet.

### Guidelines for a "Self-Contained" Question:
1.  **Contextual Setup**: Provide enough context to establish the scenario (e.g., "A developer is using the pandas Dataframe API to merge datasets"). You must identify the specific classes, methods, modules or concepts involved so the question is self-contained.
    HOWEVER, you must not explain the logic or rules of those methods (e.g., do not explain the difference between an inner or outer join, or how a specific aggregation is calculated).
    The solver should deduce the behavior based on their knowledge of the library's design.
2.  **Conceptual over Verbatim**: Test the "why" and "how" of the library's design patterns rather than asking for specific method names or definitions found in the text.
3.  **No Meta-References**: Do not use phrases like "According to the document" or "As mentioned in the snippet." as solver has no access to the document and is not assumed to remember in verbatim manner.

### Output JSON Schema
You must output a single JSON object:
{
    "question": "A comprehensive, self-contained scenario-based question including necessary context and code snippets.",
    "answer": "A detailed explanation that clarifies the mechanics and consequences of the library's behavior."
}

### Research Task
Base the task on the content of the provided file path. Use the File System MCP to examine the base class definitions or related utility functions to ensure the 'answer' is technically accurate."""
    agent = AgentWrapper[QAProblem].create(
        name="qa_generator",
        instructions=prompt,
        model="gpt-5-mini",
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
        max_turns=20,
    )
    return ret.final_output()


class QAGenerationOutput(BaseModel):
    tasks: list[QAProblem]


async def create_multiple_qas(
    local_dir: Path, file_path: str, topic: Topic, filesystem_mcp: MCPServerStdio
) -> list[QAProblem]:
    prompt = """\
You are an expert Python technical interviewer and library documentation specialist.
Your goal is to create one or more high-quality, self-contained question and answer pairs based on the provided library documentation snippet.

### Core Objectives:
- **Atomic Scope**: Each Q&A pair must focus on a single specific architectural concept or mechanic. If a document is complex, generate multiple separate Q&A pairs.
- **Identify, Don't Explain**: Provide enough context to establish the scenario (e.g., identify the library and classes), but do not give away the logic, rules, or definitions. The solver must provide the expertise.

### Guidelines for "Self-Contained" Questions:
1. **Contextual Setup**: Provide enough context to establish the scenario (e.g., "A developer is using the AST transformation API to rename columns"). You must identify the specific classes, methods, modules, or concepts involved so the question is self-contained. HOWEVER, you must not explain the logic or rules (e.g., do not explain the difference between BFS and DFS or how a specific property is calculated).
2. **Conceptual over Verbatim**: Test the "why" and "how" of the library's design patterns rather than asking for specific text found in the documentation.
3. **No Meta-References**: Do not use phrases like "According to the document" as the solver has no access to the documentation.
4. **Code-Centric**: Use code snippets to illustrate the scenario where it clarifies the technical problem.

### Examples of Desired Granularity and Depth:

**Example 1 (Focus: AST Representation vs. Runtime Logic)**
- **Question**: "A developer parses a query and inspects the representation: `Select(expressions=[Column(this=Identifier(this=a))])`. Explain the mapping between this representation and the library's internal `Expression` instance. Specifically, what does the `expressions=` label indicate relative to the instance's `args` dict, and why might some optional child keys be missing from the `repr()` output?"
- **Answer**: "The names (Select, Column) are Expression subclasses. Labels like `expressions=` are keys in the instance's `self.args` dictionary. Missing keys are omitted from `repr()` by default if they are `None` or empty lists to keep the view compact; they can be revealed using a verbose `to_s()` call."

**Example 2 (Focus: Tree Integrity during Mutation)**
- **Question**: "When mutating an AST node (e.g., using `append`, `set`, or `replace`), the library maintains parent/child links automatically. What are the consequences of manually modifying the `args` dictionary directly instead of using these methods, specifically regarding the `.parent` and `.arg_key` attributes of the child nodes?"
- **Answer**: "Directly modifying `args` bypasses the `_set_parent` helper. This leaves the child's `.parent`, `.arg_key`, and `.index` attributes incorrect or out of date, breaking bi-directional traversal and potentially leaving cached hashes invalid."

**Example 3 (Focus: Namespace Conflicts in Attribute Access)**
- **Question**: "In an interactive environment, a developer notices they can access column data via `df.A`. If a DataFrame has a column named 'count', explain why `df.count` might return a bound method instead of the column data, and what the precedence rules are for attribute-style access."
- **Answer**: "Built-in class methods and public attributes of the DataFrame object take precedence over column names. Since 'count' is a standard method for calculating non-NA cells, the attribute access resolves to the method. To access the data, the developer must use string indexing: `df['count']`."

### Output JSON Schema
You must output a single JSON object containing a list of tasks:
{
    "tasks": [
        {
            "question": "A focused, self-contained scenario-based question.",
            "answer": "A detailed explanation of the mechanics and consequences."
        }
    ]
}

### Research Task
Base the tasks on the provided file path. Use the File System MCP to verify internal mechanics (e.g., base class logic or private helpers) so the 'answer' is technically accurate regarding how the library handles its operations."""
    agent = AgentWrapper[QAGenerationOutput].create(
        name="qa_generator",
        instructions=prompt,
        model="gpt-5-mini",
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
        max_turns=20,
    )
    return ret.final_output().tasks
