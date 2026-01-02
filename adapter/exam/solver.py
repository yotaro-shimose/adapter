from typing import Self
from dataclasses import dataclass
from create_coding_exam import ExamConfig
from adapter.exam.exam import load_exam_from_csv
from oai_utils.vllm import VLLMSetup
from adapter.utils.id import gen_id
from create_coding_exam import EmptyResponseDetector
from pathlib import Path

from loguru import logger
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from adapter.exam.exam import CodingExam
from adapter.exam.renv import RustCodingEnvironment


@dataclass
class Solver:
    agent: Agent

    def solve_exam(
        self,
        exam: CodingExam,
        with_library: bool = True,
        vllm_port: int | None = None,
    ) -> bool:
        """Orchestrate the exam solving process using an AI agent."""
        logger.info(
            f"Starting exam solving (with_library={with_library}) for exam ID: {exam.id}"
        )

        # Set up a temporal repository at the problem_commit
        # We use the library info stored in the exam
        with RustCodingEnvironment(
            branch_name=gen_id(f"solve-{'with-lib' if with_library else 'no-lib'}"),
            project=exam.project,
            library=exam.library,
            image=exam.image_name,
            vllm_port=vllm_port,
        ) as env:
            # Checkout the problem commit
            logger.info(f"Checking out problem commit: {exam.problem_commit}")
            env.cloned_repo.checkout(exam.problem_commit)

            workspace = env.workspace
            empty_response_detector = EmptyResponseDetector()

            # Phase 1: Ask agent to create question, solution, and test
            conversation = Conversation(
                agent=self.agent,
                workspace=workspace,
                callbacks=[empty_response_detector],
            )
            empty_response_detector.set_conversation(conversation)

            lib_info = (
                f"The library source code is already available for your reference in `repositories/{exam.library.name}/`."
                if with_library
                else f"The library source code is NOT available for your reference. You must solve this using your internal knowledge of `{exam.library.name}`."
            )

            prompt = f"""\
    <task>
    You are a rust developer tasked with solving a coding exam.
    Here is the question:
    {exam.question}

    The project is already set up for you. 
    {lib_info}

    You should:
    1. Implement the solution in `src/lib.rs`.
    2. Run the tests in `tests/` to verify your solution (you might need to install `{exam.library.name}` dependency if you need it.
    3. Once tests pass, you are done.
    </task>
    """
            conversation.send_message(prompt)
            conversation.run()

            # Final verification with cargo test
            logger.info("Running final verification with cargo test")
            ret = env.run_test()
            return ret.is_success

    @classmethod
    def create(cls, model: str, base_url: str, api_key: str | None = None) -> Self:
        return cls(
            agent=Agent(
                llm=LLM(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                ),
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                    Tool(name=TaskTrackerTool.name),
                ],
            )
        )

    @classmethod
    def from_vllm_setup(
        cls, vllm_setup: VLLMSetup, docker_to_host: bool = True
    ) -> Self:
        if docker_to_host:
            base_url = f"http://host.docker.internal:{vllm_setup.port}/v1"
        else:
            base_url = f"http://localhost:{vllm_setup.port}/v1"
        return cls.create(
            model=vllm_setup.litellm_model(),
            base_url=base_url,
            api_key=vllm_setup.api_key,
        )


def main():
    exam_id = "exam-e1874191-36cf-420e-b991-0595eda7be86"
    config = ExamConfig.default()
    exam = load_exam_from_csv(
        Path("exams.csv"),
        exam_id=exam_id,
        image_name=config.image_name,
        project_dir=config.project_dir,
        library_dir=config.library_dir,
    )
    vllm_setup = VLLMSetup.qwen3()
    agent = Solver.from_vllm_setup(vllm_setup)
    agent.solve_exam(exam)


if __name__ == "__main__":
    main()
