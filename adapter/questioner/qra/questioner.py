from adapter.topic.filtering import is_useful_for_users
from adapter.questioner.qra.reasoner import hindsight_reasoning_retriable
from async_utils import gather_with_semaphore
from adapter.models.problems import QRA
from adapter.questioner.qra.qa import create_multiple_qas_retriable
from oai_utils.agent import AgentWrapper, AgentRunFailure, AgentsSDKModel
from adapter.models.types import ProblemType
from pydantic.main import BaseModel
from agents.mcp.server import MCPServerStdio
from adapter.topic.topics import Topic
from pathlib import Path
from loguru import logger


class DispatchResult(BaseModel):
    problem_type: ProblemType


async def dispatch_topic(topic: Topic, model: AgentsSDKModel) -> ProblemType:
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
        output_type=DispatchResult,
        model=model,
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
    local_dir: Path,
    file_path: str,
    topic: Topic,
    filesystem_mcp: MCPServerStdio,
    model: AgentsSDKModel,
) -> list[QRA] | None:
    if not await is_useful_for_users(topic, model):
        logger.info(f"Skipping topic as it is not useful for users: {topic.title}")
        return None

    # problem_type = await dispatch_topic(topic)
    problem_type: ProblemType = "qa"

    try:
        if problem_type == "qa":
            logger.info(f"Creating QA problem for topic: {topic.title}")
            qas = await create_multiple_qas_retriable(
                local_dir, file_path, topic, filesystem_mcp, model
            )
            reasonings = await gather_with_semaphore(
                [
                    hindsight_reasoning_retriable(
                        qa,
                        local_dir,
                        file_path,
                        filesystem_mcp=filesystem_mcp,
                        model=model,
                    )
                    for qa in qas
                ],
                3,
            )
            results = []
            for qa, reasoning in zip(qas, reasonings):
                if reasoning is not None:
                    results.append(
                        QRA(question=qa.question, answer=qa.answer, reasoning=reasoning)
                    )
            return results
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
    except AgentRunFailure as e:
        logger.warning(f"Agent failed to create task for topic: {topic.title}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during problem creation for topic: {topic.title}: {e}"
        )
        return None
