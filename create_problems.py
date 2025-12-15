from itertools import chain
from pathlib import Path

import httpx

from adapter.problem import ProblemDataset
from adapter.questioner import (
    Topic,
    find_topics,
    list_document_filepaths,
    questioner,
)
from adapter.utils.async_util import gather_with_semaphore
from adapter.utils.savable import Savable


class TopicEntity(Savable):
    file_path: str
    topic: Topic


class TopicEntities(Savable):
    topics: list[TopicEntity]


async def main():
    # repo_url = httpx.URL("https://github.com/pydantic/pydantic")
    repo_url = httpx.URL("https://github.com/SWE-agent/SWE-ReX/tree/main/docs")
    repo_name = str(repo_url).split("/")[-1]
    save_path = Path(f"{repo_name}_problems.json")
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
    file_paths = await list_document_filepaths(repo_url)
    # file_path = "/docs/concepts/models.md"

    if not topic_save_path.exists():
        topics = await gather_with_semaphore(
            [find_topics(repo_url, file_path) for file_path in file_paths.file_paths],
            3,
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
        file_topics.save(topic_save_path)
    else:
        file_topics = TopicEntities.load(topic_save_path)

    problems = await gather_with_semaphore(
        [
            questioner(repo_url, file_topic.file_path, file_topic.topic)
            for file_topic in file_topics.topics[:10]
        ],
        3,
    )
    dataset = ProblemDataset(problems=problems)
    dataset.save(Path("problems_10.json"))
    print(problems)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
