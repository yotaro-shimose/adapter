from adapter.topic.topics import Topic
from loguru import logger
from pydantic import BaseModel
from oai_utils.agent import AgentWrapper, AgentsSDKModel


class UsefulnessResult(BaseModel):
    is_useful: bool
    reason: str


async def is_useful_for_users(topic: Topic, model: AgentsSDKModel) -> bool:
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
6. Can not be tested as a form of programming exercise.

A topic IS useful if it describes:
1. Features, functions, classes, modules or concepts of the library.
2. How to use specific parts of the library.
3. Tutorials, examples, or guides for users.

### Output JSON Schema
{
    "is_useful": "boolean. True if the topic is useful for library learners, False otherwise.",
    "reason": "A brief explanation of why the topic is or is not useful."
}""",
        output_type=UsefulnessResult,
        model=model,
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
