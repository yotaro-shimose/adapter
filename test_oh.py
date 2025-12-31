from adapter.utils.savable import Savable
from openhands.sdk.context.condenser.llm_summarizing_condenser import (
    LLMSummarizingCondenser,
)
import uuid
from typing import Self
from pathlib import Path
from pydantic.main import BaseModel
import os
import tempfile
import shutil
import subprocess
import platform
from dotenv import load_dotenv
from loguru import logger

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace


class TemporalCodingRepositoryError(Exception):
    """Custom exception for TemporalCodingRepository errors."""

    pass


def gen_id(prefix: str):
    return f"{prefix}-{uuid.uuid4()}"


class GitRepository(BaseModel):
    name: str
    local_dir: Path

    def model_post_init(self, __context) -> None:
        """Verify the directory exists and is a valid git repository."""
        if not self.local_dir.exists():
            raise TemporalCodingRepositoryError(
                f"Repository directory does not exist: {self.local_dir}"
            )
        # Check if it's a valid git repo
        self.run_git(["rev-parse", "--is-inside-work-tree"])

    @classmethod
    def create(cls, name: str, local_dir: Path) -> Self:
        return cls(name=name, local_dir=local_dir)

    def run_git(self, args: list[str], cwd: Path | None = None) -> str:
        command = ["git"] + args
        working_dir = cwd or self.local_dir
        logger.debug(f"Running git command: {' '.join(command)} in {working_dir}")
        try:
            result = subprocess.run(
                command,
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            msg = f"Git command failed in repository '{self.name}': {e.stderr or e.stdout}"
            logger.error(msg)
            raise TemporalCodingRepositoryError(msg) from e

    def checkout(self, branch: str, create: bool = False) -> None:
        args = ["checkout", "-b", branch] if create else ["checkout", branch]
        self.run_git(args)
        self.chmod_777()

    def add(self, path: str = ".") -> None:
        self.chmod_777()
        self.run_git(["add", path])

    def commit(self, message: str) -> None:
        self.run_git(["commit", "-m", message])

    def push(self, remote: str, branch: str) -> None:
        self.run_git(["push", remote, branch])

    def rev_parse(self, ref: str = "HEAD") -> str:
        return self.run_git(["rev-parse", ref])

    @property
    def exists(self) -> bool:
        return self.local_dir.exists()

    def chmod_777(self) -> None:
        """Apply chmod -R 777 to the repository directory."""
        logger.debug(f"Applying chmod -R 777 to {self.local_dir}")
        try:
            subprocess.run(["chmod", "-R", "777", str(self.local_dir)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply chmod -R 777: {e.stderr or e.stdout}")


class CodingExam(Savable):
    id: str
    image_name: str
    project: GitRepository
    library: GitRepository
    solution_commit: str
    problem_commit: str
    question: str


class TemporalCodingRepository(BaseModel):
    branch_name: str
    project: GitRepository
    library: GitRepository
    cloned_repo: GitRepository | None = None

    @classmethod
    def create(
        cls, project: GitRepository, library: GitRepository, branch_name: str
    ) -> Self:
        return cls(branch_name=branch_name, library=library, project=project)

    @property
    def _repo(self) -> GitRepository:
        if self.cloned_repo is None:
            raise TemporalCodingRepositoryError(
                "cloned_repo is not set. Did you call setup()?"
            )
        return self.cloned_repo

    def setup(self, setup_library: bool = True) -> Self:
        """Create a new coding environment for AI agent."""
        logger.info(
            f"Setting up temporal repository {self.branch_name} for project {self.project.name} (setup_library={setup_library})"
        )
        temp_dir = Path(tempfile.mkdtemp())

        # 1. We clone the root project dir into tempdir.
        logger.debug(f"Cloning project {self.project.local_dir} to {temp_dir}")
        try:
            subprocess.run(
                ["git", "clone", str(self.project.local_dir), str(temp_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise TemporalCodingRepositoryError(
                f"Initial clone failed: {e.stderr or e.stdout}"
            ) from e

        # Now that it's a valid repo, we can instantiate the GitRepository object
        self.cloned_repo = GitRepository(
            name=f"{self.project.name}-cloned", local_dir=temp_dir
        )
        logger.info(f"Created temporal repository at: {self.cloned_repo.local_dir}")

        self._create_branch()
        if setup_library:
            self._setup_library()

        # Fix permissions for Docker mount (OpenHands user UID 10001)
        self.cloned_repo.chmod_777()

        logger.success(f"Temporal repository {self.branch_name} setup complete")
        return self

    def _create_branch(self) -> None:
        """2. We create a new git branch in the cloned repository."""
        self._repo.checkout(self.branch_name, create=True)

    def _setup_library(self) -> None:
        """3. We clone reference library to PROJECT_ROOT/repositories/{library_name}"""
        repo_dir = self._repo.local_dir / "repositories" / self.library.name
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", str(self.library.local_dir), str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise TemporalCodingRepositoryError(
                f"Library clone failed: {e.stderr or e.stdout}"
            ) from e

    def push_exam(self, message: str, run_tests: bool = True) -> str:
        """Commit changes, push to original project, and return commit hash."""
        logger.info(f"Pushing coding exam commit: {message} ({self.branch_name})")
        # 1. Check for changes
        self._repo.add()
        status = self._repo.run_git(["status", "--porcelain"])
        if not status:
            logger.warning("No changes detected in the repository")
            # If no changes, still return the current commit hash
            return self._repo.rev_parse()

        # 2. Verify with cargo test if requested
        if run_tests:
            logger.debug("Running cargo test to verify solution")
            try:
                subprocess.run(
                    ["cargo", "test"],
                    cwd=self._repo.local_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                msg = f"Cargo test failed: {e.stderr or e.stdout}"
                logger.error(msg)
                raise TemporalCodingRepositoryError(msg) from e

        # 3. Commit changes
        logger.debug(f"Committing changes: {message}")
        self._repo.commit(message)

        # 4. Push branch to origin
        logger.debug(f"Pushing branch {self.branch_name} to origin")
        self._repo.push("origin", self.branch_name)

        # 5. Get the latest commit hash
        commit_hash = self._repo.rev_parse()
        logger.success(f"Commit pushed: {commit_hash}")
        return commit_hash

    def cleanup(self) -> None:
        """Clean up the coding environment."""
        try:
            if self.cloned_repo and self.cloned_repo.exists:
                logger.info(
                    f"Cleaning up temporal repository at {self.cloned_repo.local_dir}"
                )
                shutil.rmtree(self.cloned_repo.local_dir)
        except Exception as e:
            msg = f"Cleanup failed: {e}"
            logger.error(msg)
            raise TemporalCodingRepositoryError(msg) from e


def detect_platform() -> str:
    """Detects the correct Docker platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def generate_exam(
    project: GitRepository,
    library: GitRepository,
    image_name: str,
    agent: Agent,
) -> CodingExam:
    """Orchestrate the exam creation process using an AI agent."""
    logger.info(
        f"Starting exam generation for project: {project.name}, library: {library.name}"
    )
    base_id = str(uuid.uuid4())
    repo = TemporalCodingRepository.create(
        project, library, branch_name=f"repo-{base_id}"
    )
    repo.setup()

    try:
        with DockerWorkspace(
            server_image=image_name,
            platform=detect_platform(),
            mount_dir=str(repo._repo.local_dir.absolute()),
            forward_env=[
                "GOOGLE_API_KEY",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
            ],
        ) as workspace:
            # Phase 1: Ask agent to create question, solution, and test
            conversation = Conversation(agent=agent, workspace=workspace)
            prompt = f"""\
<task>
You are a rust coding exam generator for library `{library.name}`.
You are inside a cargo project. We cloned `{library.name}` repository in the at repositories/{library.name}/.
You should first read the `{library.name}` repository.
Then 
1. Create a question about usage of the library in README.md in our project root.
2. Create a solution in lib.rs file.
3. Write down a test code for the solution.
4. Confirm your test passes for your solution.
</task>

<Guidelines>
- When installing the dependencies, do not use path dependency. You should install it via cargo command. The library repository is just for your reference.
- How to install the library is not part of the exam.
- Do not refer to the documents/source code of `{library.name}` repository in your question.
    - The `{library.name}` repository is not visible to solver. Solver is expected to remember the usage of the library.
    - For example testing solver's understanding about specific functions is good, but asking solver to read the source code is not good.
- The problems should test the practical usage of multiple features combined, not just a single feature.
- Tests should be located in tests/ directory which will be hidden from solver.
</Guidelines>
"""
            conversation.send_message(prompt)
            conversation.run()

            # Commit and push solution
            solution_commit_hash = repo.push_exam(
                message=f"feat: coding exam solution {repo.branch_name}"
            )

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
        problem_commit_hash = repo.push_exam(
            message=f"feat: coding exam problem {repo.branch_name}", run_tests=False
        )

        # Read the generated question
        readme_path = repo._repo.local_dir / "README.md"
        question = readme_path.read_text() if readme_path.exists() else ""

        return CodingExam(
            id=f"exam-{base_id}",
            image_name=image_name,
            project=project,
            library=library,
            solution_commit=solution_commit_hash,
            problem_commit=problem_commit_hash,
            question=question,
        )
    finally:
        repo.cleanup()


def solve_exam(
    exam: CodingExam,
    agent: Agent,
    with_library: bool = True,
) -> bool:
    """Orchestrate the exam solving process using an AI agent."""
    logger.info(
        f"Starting exam solving (with_library={with_library}) for exam ID: {exam.id}"
    )

    # Set up a temporal repository at the problem_commit
    # We use the library info stored in the exam
    repo = TemporalCodingRepository.create(
        exam.project,
        exam.library,
        branch_name=f"solve-{exam.id}-{'with-lib' if with_library else 'no-lib'}",
    )
    repo.setup(setup_library=with_library)

    # Checkout the problem commit
    logger.info(f"Checking out problem commit: {exam.problem_commit}")
    repo._repo.checkout(exam.problem_commit)

    try:
        with DockerWorkspace(
            server_image=exam.image_name,
            platform=detect_platform(),
            mount_dir=str(repo._repo.local_dir.absolute()),
            forward_env=[
                "GOOGLE_API_KEY",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
            ],
        ) as workspace:
            conversation = Conversation(agent=agent, workspace=workspace)

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
2. Run the tests in `tests/` to verify your solution (you might need to install `{exam.library.name}` dependency if you need it, but the library repository itself is hidden from solver if mentioned above).
3. Once tests pass, you are done.
</task>
"""
            conversation.send_message(prompt)
            conversation.run()

        # Final verification with cargo test
        logger.info("Running final verification with cargo test")
        try:
            subprocess.run(
                ["cargo", "test"],
                cwd=repo._repo.local_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.success(f"Exam {exam.id} solved successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Final verification failed: {e.stderr or e.stdout}")
            return False
    finally:
        repo.cleanup()


if __name__ == "__main__":
    load_dotenv()
    model_name = "gemini/gemini-3-pro-preview"
    # model_name = "gemini/gemini-3-flash-preview"
    out_path = Path("exam.json")

    llm = LLM(
        model=model_name,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    condenser = LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_tokens=150_000,
        keep_first=3,
    )

    agent_instance = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
        condenser=condenser,
    )

    example_project = GitRepository.create(
        name="rust-benchmarks",
        local_dir=Path("../rust-benchmarks").absolute(),
    )
    example_library = GitRepository.create(
        name="inturn", local_dir=Path("repositories/inturn").absolute()
    )

    if not out_path.exists():
        exam = generate_exam(
            project=example_project,
            library=example_library,
            image_name="ohserver-rust",
            agent=agent_instance,
        )
        exam.save(out_path)
        logger.info(f"Generated Exam ID: {exam.id}")
        logger.info(f"Solution Commit: {exam.solution_commit}")
        logger.info(f"Problem Commit: {exam.problem_commit}")
        logger.debug(f"Question Preview: {exam.question[:100]}...")
        logger.info(f"Exam saved: {out_path}")
    else:
        exam = CodingExam.load(out_path)
        logger.info(f"Loaded Exam ID: {exam.id}")
        logger.info(f"Solution Commit: {exam.solution_commit}")
        logger.info(f"Problem Commit: {exam.problem_commit}")
        logger.debug(f"Question Preview: {exam.question[:100]}...")

    # Demonstrate solving the exam in dual-mode
    logger.info("--- Starting Dual-Mode Solver Phase ---")

    # # 1. Without library source code (Black box)
    # logger.info("Setting 1: Solving WITHOUT library source code access")
    # solved_no_lib = solve_exam(exam, agent_instance, with_library=False)

    # 2. With library source code (White box)
    logger.info("Setting 2: Solving WITH library source code access")
    solved_with_lib = solve_exam(exam, agent_instance, with_library=True)

    # if solved_no_lib:
    #     logger.success("Solver successfully completed the exam WITHOUT library source!")
    # else:
    #     logger.error("Solver failed to complete the exam WITHOUT library source.")

    if solved_with_lib:
        logger.success("Solver successfully completed the exam WITH library source!")
    else:
        logger.error("Solver failed to complete the exam WITH library source.")

    logger.success("All done!")
