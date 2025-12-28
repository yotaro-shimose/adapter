from openai.types.shared.reasoning import Reasoning
from agents.model_settings import ModelSettings
from agents.tool import WebSearchTool
from dotenv.main import load_dotenv
import asyncio
from oai_utils.client import get_aoai
from oai_utils.mcp.github import github_mcp
from oai_utils.agent import AgentWrapper

import logfire


logfire.configure()
logfire.instrument_openai()  # instrument all OpenAI clients globally


async def main():
    load_dotenv()
    async with github_mcp() as gh:
        agent = AgentWrapper[str].create(
            name="whatever",
            instructions="""\
Please answer user question about coding.
You can also use the web search tool to search about libraries.
You are able to use the github functions to get the information about source codes.
    """,
            mcp_servers=[gh],
            model=get_aoai("gpt-5.2"),
            tools=[WebSearchTool()],
            model_settings=ModelSettings(
                max_tokens=400000, reasoning=Reasoning(effort="minimal")
            ),
        )
        ret = await agent.run(
            input="""\
## 2025 Data Engineering Exam: Complex Document Parsing

### Problem Statement

You are building an automated accounting tool. You must create an asynchronous Python function that extracts data from an OpenAI receipt and prepares a landing table in a SQL database.

Unlike simple files, this receipt contains data in **multiple layouts**:

- **Key-Value Pairs:** (e.g., "Invoice number", "Date paid").
- **Line Items Table:** (e.g., "Description", "Qty", "Tax").
- **Calculated Totals:** Including tax and JCT (Japan Consumption Tax).


### Your Task

Write a function `generate_receipt_schema` that performs the following:

1. **Dynamic Extraction:** Use **Kreuzberg** to extract the text content.
2. **Schema Inference:** Since the document contains multiple tables, your code must dynamically identify unique headers (e.g., `Description`, `Amount paid`, `Receipt number`).
3. **AST Construction with `sqlglot`:** Create a `CREATE TABLE` expression.
4. **Constraint:** You must use `sqlglot.parse_identifier` for every column name to handle spaces and special characters like `JCT Japan (10% on $100.00)`.
5. **Constraint:** You must use `sqlglot.exp.DataType` to ensure the "Amount" columns are handled correctly.
6. **Multi-Dialect Output:** Ensure the final SQL string is generated for the `target_dialect` provided.
"""
        )
        print(ret.final_output())


if __name__ == "__main__":
    asyncio.run(main())
