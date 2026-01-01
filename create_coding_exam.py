import asyncio
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Self

import polars as pl
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from loguru import logger
from openhands.sdk import LLM, Agent, Conversation, Event, Tool
from openhands.sdk.conversation.base import BaseConversation
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from pydantic.main import BaseModel

from adapter.exam.exam import CodingExam
from adapter.exam.renv import RustCodingEnvironment
from adapter.exam.repository import GitRepository
from adapter.find_topic import find_topics
from adapter.questioner.qra.finder import list_document_filepaths
from adapter.topic.filtering import is_useful_for_users
from adapter.topic.topics import TopicEntities, TopicEntity
from async_utils import gather_with_semaphore


class Config(BaseModel):
    model_name: str
    image_name: str
    project_dir: Path
    library_dir: Path
    topic_extraction_semaphore: int
    exam_generation_semaphore: int
    max_topics: int
    batch_size: int
    output_file: Path
    topics_file: Path

    @classmethod
    def default(cls) -> Self:
        return cls(
            model_name="gemini/gemini-3-flash-preview",
            # model_name="gemini/gemini-3-pro-preview",
            image_name="ohserver-rust",
            project_dir=Path("../rust-benchmarks").absolute(),
            library_dir=Path("repositories/numrs").absolute(),
            topic_extraction_semaphore=3,
            exam_generation_semaphore=5,
            max_topics=3,
            batch_size=30,
            output_file=Path("exams.csv"),
            topics_file=Path("topics.json"),
        )


def gen_id(prefix: str):
    return f"{prefix}-{uuid.uuid4()}"


def detect_platform() -> str:
    """Detects the correct Docker platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


@dataclass
class EmptyResponseDetector:
    counts: int = 0
    limit: int = 1
    conversation: BaseConversation | None = None

    def set_conversation(self, conversation: BaseConversation) -> None:
        self.conversation = conversation

    def __call__(self, event: Event) -> None:
        if self.conversation is None:
            raise ValueError("The EmptyResponseDetector is not initialized yet used")

        if "[no text content]" in str(event.visualize):
            self.counts += 1
            if self.counts > self.limit:
                self.conversation.pause()
                raise ValueError("Looks like agent is stuck")


def generate_exam(
    project: GitRepository,
    library: GitRepository,
    agent: Agent,
    image_name: str,
    topic: TopicEntity,
) -> CodingExam | None:
    """Orchestrate the exam creation process using an AI agent."""
    logger.info(
        f"Starting exam generation for project: {project.name}, library: {library.name}, topic: {topic.topic.title}"
    )
    with RustCodingEnvironment(
        branch_name=gen_id("repo"),
        project=project,
        library=library,
        image=image_name,
    ) as env:
        # Use auto-mounting from RustCodingEnvironment
        workspace = env.workspace

        empty_response_detector = EmptyResponseDetector()

        # Phase 1: Ask agent to create question, solution, and test
        conversation = Conversation(
            agent=agent, workspace=workspace, callbacks=[empty_response_detector]
        )
        empty_response_detector.set_conversation(conversation)
        prompt = f"""\
<task>
You are a rust coding exam generator for library `{library.name}`.
You are inside a cargo project. We cloned `{library.name}` repository in the at repositories/{library.name}/.
You should first read the `{library.name}` repository.

Target Topic: {topic.topic.title}
Topic Description: {topic.topic.description}
Reference File: {topic.file_path}

Then 
1. Create a question about usage of the library regarding the target topic.
2. Create a solution in lib.rs file.
3. Write down a test code for the solution.
4. Confirm your test passes for your solution.
</task>

<Guidelines>
- When installing the dependencies, do not use path dependency.
    - You should install it via cargo command. The library repository is just for your reference.
    - How to install the library is not part of the exam.
- Do not refer to the documents/source code of `{library.name}` repository in your question.
    - The `{library.name}` repository is not visible to solver. Solver is expected to remember the usage of the library.
    - For example testing solver's understanding about specific functions is good, but asking solver to read the source code is not good.
- Tests should be located in tests/ directory which will be hidden from solver.
</Guidelines>
"""
        conversation.send_message(prompt)
        conversation.run()

        # Commit and push solution
        solution_commit_hash = env.push_exam(
            message=f"feat: coding exam solution {env.branch_name}"
        )
        if not solution_commit_hash:
            logger.error("Failed to push solution commit")
            return None

        # Phase 2: Ask agent to clean the solution
        clean_prompt = """\
<task>
Now, please clean the solution from lib.rs. 
The goal is to leave the lib.rs in a state where the solver needs to implement the solution, but the tests you wrote in tests/ directory still exist and can be used to verify the solver's work.
You should remove the implementation logic and also remove all imports. The solver is expected to know which modules and types they need to import.
Only leave empty function signatures if they are strictly necessary for the tests to even start compiling, but if possible, let the solver define them as well.
</task>
"""
        conversation.send_message(clean_prompt)
        conversation.run()

        # Commit and push problem (without running tests, as it's now "broken" by design)
        problem_commit_hash = env.push_exam(
            message=f"feat: coding exam problem {env.branch_name}", run_tests=False
        )
        if not problem_commit_hash:
            logger.error("Failed to push problem commit")
            return None

        # Read the generated question
        readme_path = env.cloned_repo.local_dir / "README.md"
        question = readme_path.read_text() if readme_path.exists() else ""

        return CodingExam(
            id=gen_id("exam"),
            image_name=image_name,
            project=project,
            library=library,
            solution_commit=solution_commit_hash,
            problem_commit=problem_commit_hash,
            question=question,
        )


async def async_main():
    load_dotenv()
    config = Config.default()

    # Initialize basic components
    llm = LLM(
        model=config.model_name,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    litellm = LitellmModel(
        model=config.model_name, api_key=os.getenv("LITELLM_API_KEY")
    )

    # 1. Topic Extraction
    logger.info(f"Topic extraction for repository: {config.library_dir.name}")
    topic_save_path = (
        config.project_dir / "topics.json"
    )  # Adjust path as needed, using project dir for now or create data dir

    if not topic_save_path.exists():
        logger.info("Topic file not found, extracting topics from documents")
        file_paths = await list_document_filepaths(config.library_dir, model=litellm)
        logger.info(f"Found {len(file_paths.file_paths)} document files")
        topics = await gather_with_semaphore(
            [
                find_topics(config.library_dir, file_path, model=litellm)
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
            f"Extracted {len(file_topics.topics)} topics, saving to {topic_save_path}"
        )
        file_topics.save(topic_save_path)
    else:
        logger.info(f"Loading existing topics from {topic_save_path}")
        file_topics = TopicEntities.load(topic_save_path)

    # 2. Exam Generation
    logger.info(f"Generating exams for {len(file_topics.topics)} topics")

    example_project = GitRepository(
        name="rust-benchmarks",
        local_dir=config.project_dir,
    )
    example_library = GitRepository(
        name=config.library_dir.name, local_dir=config.library_dir
    )

    exams_data = []
    exams_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(config.exam_generation_semaphore)

    async def process_topic(topic: TopicEntity):
        is_useful = await is_useful_for_users(topic.topic, model=litellm)
        if not is_useful:
            return
        async with semaphore:
            try:
                # Create a fresh agent for each exam to avoid state pollution
                agent_instance = Agent(
                    llm=llm,
                    tools=[
                        Tool(name=TerminalTool.name),
                        Tool(name=FileEditorTool.name),
                        Tool(name=TaskTrackerTool.name),
                    ],
                )

                # Run the sync generate_exam in a thread
                exam = await asyncio.to_thread(
                    generate_exam,
                    project=example_project,
                    library=example_library,
                    agent=agent_instance,
                    image_name=config.image_name,
                    topic=topic,
                )

                if exam is None:
                    logger.warning(
                        f"Failed to generate exam for topic {topic.topic.title}"
                    )
                    return

                async with exams_lock:
                    exams_data.append(
                        {
                            "id": exam.id,
                            "topic_title": topic.topic.title,
                            "topic_description": topic.topic.description,
                            "file_path": topic.file_path,
                            "question": exam.question,
                            "solution_commit": exam.solution_commit,
                            "problem_commit": exam.problem_commit,
                            "image_name": config.image_name,
                            "status": "generated",
                        }
                    )

                    # Periodic save
                    if len(exams_data) % config.batch_size == 0:
                        logger.info(
                            f"Progress: {len(exams_data)} exams generated. Saving intermediate results..."
                        )
                        df = pl.DataFrame(exams_data)
                        df.write_csv(config.output_file)

            except Exception as e:
                logger.error(f"Error processing topic {topic.topic.title}: {e}")
                async with exams_lock:
                    exams_data.append(
                        {
                            "id": str(uuid.uuid4()),  # Placeholder ID
                            "topic_title": topic.topic.title,
                            "topic_description": topic.topic.description,
                            "file_path": topic.file_path,
                            "question": "",
                            "solution_commit": "",
                            "problem_commit": "",
                            "image_name": config.image_name,
                            "status": f"failed: {e}",
                        }
                    )

    tasks = [process_topic(topic) for topic in file_topics.topics[: config.max_topics]]
    await gather_with_semaphore(
        tasks, config.exam_generation_semaphore, progressbar=True
    )  # Reusing gather for progress bar

    # Final save
    df = pl.DataFrame(exams_data)
    df.write_csv(config.output_file)
    logger.success(f"Saved {len(exams_data)} exams to {config.output_file}")


if __name__ == "__main__":
    asyncio.run(async_main())
