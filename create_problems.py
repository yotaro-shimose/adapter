from adapter.questioner.questioner import questioner
from adapter.models.problems import QRADataset, QRA
from adapter.questioner.finder import list_document_filepaths
from itertools import chain
from pathlib import Path
import asyncio

from dotenv.main import load_dotenv
from loguru import logger
from tqdm.asyncio import tqdm_asyncio
from oai_utils.mcp.filesystem import filesystem_mcp

from adapter.find_topic import find_topics
from adapter.topic.topics import TopicEntities, TopicEntity
from adapter.utils.async_util import gather_with_semaphore
from adapter.models.config import ProblemCreationConfig


async def main():
    load_dotenv()
    config = ProblemCreationConfig(
        repo_path=Path("./sqlglot"),
        topic_extraction_semaphore=3,
        question_generation_semaphore=30,
        max_topics=1000,
        batch_size=30,
        output_dir=Path("./data"),
        model="gpt-4o",
    )

    logger.info(f"Starting problem creation for repository: {config.repo_name}")
    logger.debug(f"Topic save path: {config.topic_save_path}")

    if not config.topic_save_path.exists():
        logger.info("Topic file not found, extracting topics from documents")
        file_paths = await list_document_filepaths(config.repo_path)
        logger.info(f"Found {len(file_paths.file_paths)} document files")
        topics = await gather_with_semaphore(
            [
                find_topics(config.repo_path, file_path)
                for file_path in file_paths.file_paths
            ],
            config.topic_extraction_semaphore,
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
            f"Extracted {len(file_topics.topics)} topics, saving to {config.topic_save_path}"
        )
        file_topics.save(config.topic_save_path)
    else:
        logger.info(f"Loading existing topics from {config.topic_save_path}")
        file_topics = TopicEntities.load(config.topic_save_path)

    logger.info(f"Generating problems for {len(file_topics.topics)} topics")
    problems: list[QRA] = []
    completed_count = 0
    save_lock = asyncio.Lock()

    async with filesystem_mcp(
        allowed_directories=[str(config.repo_path)], read_only=True
    ) as filesystem:
        semaphore = asyncio.Semaphore(config.question_generation_semaphore)

        async def process_topic(file_topic: TopicEntity):
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await questioner(
                        config.repo_path,
                        file_topic.file_path,
                        file_topic.topic,
                        filesystem_mcp=filesystem,
                        model=config.model,
                    )
                    if result:
                        async with save_lock:
                            problems.extend(result)
                except Exception as e:
                    logger.error(
                        f"Error processing topic {file_topic.topic.title}: {e}"
                    )
                finally:
                    async with save_lock:
                        completed_count += 1
                        if completed_count % 30 == 0:
                            logger.info(
                                f"Progress: {completed_count}/{config.max_topics}. Saving intermediate results..."
                            )
                            dataset = QRADataset(problems=problems)
                            dataset.save(config.output_path)

        tasks = [
            process_topic(file_topic)
            for file_topic in file_topics.topics[: config.max_topics]
        ]
        await tqdm_asyncio.gather(*tasks)

    # Final save
    dataset = QRADataset(problems=problems)
    dataset.save(config.output_path)
    logger.success(f"Saved {len(problems)} problems to {config.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
