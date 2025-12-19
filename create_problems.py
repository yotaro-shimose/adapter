from adapter.models.problems import ProblemDataset
from adapter.questioner.questioner import questioner
from adapter.questioner.finder import list_document_filepaths
from adapter.models.problems import VerifiableProblem
from adapter.models.problems import QAProblem
from itertools import chain
from pathlib import Path

from dotenv.main import load_dotenv
from loguru import logger
from more_itertools import chunked
from oai_utils.mcp.filesystem import filesystem_mcp

from adapter.find_topic import find_topics
from adapter.models.topics import TopicEntities, TopicEntity
from adapter.utils.async_util import gather_with_semaphore


async def main():
    load_dotenv()
    # repo_url = httpx.URL("https://github.com/SWE-agent/SWE-ReX/tree/main/docs")
    # repo_url = httpx.URL("https://github.com/tobymao/sqlglot")
    cloned_repo_path = Path("./sqlglot")
    repo_name = str(cloned_repo_path).split("/")[-1]
    output_path = Path(f"{repo_name}_problems.json")
    logger.info(f"Starting problem creation for repository: {repo_name}")
    # await create_and_save_problems(
    #     repo_url=repo_url,
    #     document_paths=document_paths,
    #     save_path=save_path,
    # )

    # dataset = ProblemDataset.load(save_path)
    # for i, problem in enumerate(dataset.problems):
    #     print(f"Problem {i + 1}:")
    #     print(problem.as_md())
    #     print("\n" + "=" * 40 + "\n")
    topic_save_path = Path(f"{repo_name}_topics.json")
    logger.debug(f"Topic save path: {topic_save_path}")

    # file_path = "/docs/concepts/models.md"

    if not topic_save_path.exists():
        logger.info("Topic file not found, extracting topics from documents")
        file_paths = await list_document_filepaths(cloned_repo_path)
        logger.info(f"Found {len(file_paths.file_paths)} document files")
        topics = await gather_with_semaphore(
            [
                find_topics(cloned_repo_path, file_path)
                for file_path in file_paths.file_paths
            ],
            3,
            progressbar=True,
        )
        file_topics = TopicEntities(
            topics=list(
                chain.from_iterable(
                    [
                        [
                            TopicEntity(file_path=file_path, topic=topic)
                            for topic in topics.topics
                        ]
                        for file_path, topics in zip(file_paths.file_paths, topics)
                    ]
                )
            )
        )
        logger.info(
            f"Extracted {len(file_topics.topics)} topics, saving to {topic_save_path}"
        )
        file_topics.save(topic_save_path)
    else:
        logger.info(f"Loading existing topics from {topic_save_path}")
        file_topics = TopicEntities.load(topic_save_path)

    logger.info(f"Generating problems for {len(file_topics.topics)} topics")
    problems: list[QAProblem | VerifiableProblem] = []
    async with filesystem_mcp(
        allowed_directories=[str(cloned_repo_path)], read_only=True
    ) as filesystem:
        for file_topics_batch in chunked(file_topics.topics[:10], 10):
            batch_problems = await gather_with_semaphore(
                [
                    questioner(
                        # create_qa(
                        cloned_repo_path,
                        file_topic.file_path,
                        file_topic.topic,
                        filesystem_mcp=filesystem,
                    )
                    for file_topic in file_topics_batch
                ],
                10,
                progressbar=True,
            )
            for problem_list in batch_problems:
                if problem_list:
                    problems.extend(problem_list)

            dataset = ProblemDataset(problems=problems)
            dataset.save(output_path)

    logger.success(f"Saved {len(problems)} problems to {output_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
