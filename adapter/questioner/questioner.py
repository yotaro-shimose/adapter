from adapter.questioner.reasoner import hindsight_reasoning
from adapter.utils.async_util import gather_with_semaphore
from adapter.models.problems import QRA
from adapter.questioner.qa import create_multiple_qas
from oai_utils.agent import AgentRunFailure
from adapter.questioner.coding import ProblemVerificationError
from oai_utils.agent import AgentWrapper
from adapter.models.types import ProblemType
from pydantic.main import BaseModel
from agents.mcp.server import MCPServerStdio
from adapter.models.topics import Topic
from pathlib import Path
from loguru import logger


class DispatchResult(BaseModel):
    problem_type: ProblemType


class UsefulnessResult(BaseModel):
    is_useful: bool
    reason: str


async def is_useful_for_users(topic: Topic) -> bool:
    agent = AgentWrapper[UsefulnessResult].create(
        name="topic_usefulness_checker",
        instructions="""\
You are an expert educator and technical writer.
Your task is to determine if a given topic from library documentation is useful for library learners (users who want to learn how to use the library).

A topic is NOT useful if it is about:
1. How to contribute to the library (e.g., development setup, pull request guidelines).
2. LICENSE information.
3. Changelogs or release notes.
4. Internal architecture that users don't need to know.
5. Administrative or project management details.

A topic IS useful if it describes:
1. Features, functions, classes, modules or concepts of the library.
2. How to use specific parts of the library.
3. Tutorials, examples, or guides for users.

### Output JSON Schema
{
    "is_useful": "boolean. True if the topic is useful for library learners, False otherwise.",
    "reason": "A brief explanation of why the topic is or is not useful."
}""",
        model="gpt-5-mini",
        output_type=UsefulnessResult,
    )
    ret = await agent.run(
        input=f"""\
Based on the following topic title and description, determine if it is useful for library learners.
Topic title: {topic.title}
Topic description: {topic.description}""",
    )
    result = ret.final_output()
    logger.info(
        f"Topic '{topic.title}' usefulness: {result.is_useful}. Reason: {result.reason}"
    )
    return result.is_useful


async def dispatch_topic(topic: Topic) -> ProblemType:
    agent = AgentWrapper[DispatchResult].create(
        name="problem_type_dispatcher",
        instructions="""\
You are an expert educational content designer.
Your task is to determine the most suitable type of assessment for a given topic from library documentation.
You must choose between two types of assessments: "programming" or "qa".
### Output JSON Schema
{
    "problem_type": "Either 'programming' or 'qa'. Choose 'programming' if the topic involves coding tasks, algorithms, or implementation details. Choose 'qa' for conceptual questions, definitions, or explanations."
}

### Guidelines
1. **Programming**: Choose this type if the topic's understanding can be effectively assessed through coding tasks (e.g., implementing functions, algorithms, or using specific library features).
2. **QA**: Choose this type for conceptual questions, definitions, or explanations.""",
        model="gpt-5-mini",
        output_type=DispatchResult,
    )
    ret = await agent.run(
        input=f"""\
Based on the following topic from library documentation, determine the most suitable type of assessment.
Topic title: {topic.title}
Topic description: {topic.description}""",
    )
    result = ret.final_output()
    return result.problem_type


async def questioner(
    local_dir: Path, file_path: str, topic: Topic, filesystem_mcp: MCPServerStdio
) -> list[QRA] | None:
    if not await is_useful_for_users(topic):
        logger.info(f"Skipping topic as it is not useful for users: {topic.title}")
        return None

    # problem_type = await dispatch_topic(topic)
    problem_type: ProblemType = "qa"

    try:
        if problem_type == "qa":
            logger.info(f"Creating QA problem for topic: {topic.title}")
            qas = await create_multiple_qas(local_dir, file_path, topic, filesystem_mcp)
            reasonings = await gather_with_semaphore(
                [hindsight_reasoning(qa) for qa in qas], 3
            )
            return [
                QRA(question=qa.question, answer=qa.answer, reasoning=reasoning)
                for qa, reasoning in zip(qas, reasonings)
            ]
        # elif problem_type == "programming":
        #     logger.info(f"Creating coding task for topic: {topic.title}")
        #     return [
        #         await create_coding_task(local_dir, file_path, topic, filesystem_mcp)
        #     ]
        else:
            logger.error(
                f"Unknown problem type '{problem_type}' for topic: {topic.title}"
            )
            return None
    except ProblemVerificationError:
        logger.warning(f"Failed to create a valid task for topic: {topic.title}")
        return None
    except AgentRunFailure as e:
        logger.warning(f"Agent failed to create task for topic: {topic.title}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during problem creation for topic: {topic.title}: {e}"
        )
        return None
