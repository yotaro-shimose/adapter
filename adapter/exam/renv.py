import shutil
import subprocess
import tempfile
import platform
from contextlib import ExitStack
from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from adapter.exam.exception import TemporalCodingRepositoryError
from adapter.exam.repository import GitRepository
from adapter.exam.workspace import MountableDockerWorkspace


class TestResult(BaseModel):
    is_success: bool
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


class TemporalCodingRepository(BaseModel):
    branch_name: str
    project: GitRepository
    library: GitRepository
    cloned_repo: GitRepository | None = None

    def __enter__(self) -> Self:
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    @property
    def local_dir(self) -> Path:
        return self._git.local_dir

    @property
    def _git(self) -> GitRepository:
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
        self._git.checkout(self.branch_name, create=True)

    def _setup_library(self) -> None:
        """3. We clone reference library to PROJECT_ROOT/repositories/{library_name}"""
        repo_dir = self.local_dir / "repositories" / self.library.name
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


class RustCodingEnvironment(BaseModel):
    project: GitRepository
    library: GitRepository
    branch_name: str
    image: str
    extra_mounts: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    vllm_port: int | None = None

    # Internal state
    _temp_repo: TemporalCodingRepository | None = None
    _workspace: MountableDockerWorkspace | None = None
    _exit_stack: ExitStack = PrivateAttr(default_factory=ExitStack)

    @property
    def cloned_repo(self) -> GitRepository:
        if self._temp_repo is None or self._temp_repo.cloned_repo is None:
            raise TemporalCodingRepositoryError("Environment not set up")
        return self._temp_repo.cloned_repo

    def __enter__(self) -> Self:
        logger.info(f"Setting up RustCodingEnvironment for {self.project.name}")

        # 1. Setup TemporalCodingRepository
        self._temp_repo = TemporalCodingRepository(
            branch_name=self.branch_name,
            project=self.project,
            library=self.library,
        )
        self._exit_stack.enter_context(self._temp_repo)

        # 2. Prepare mounts
        mounts = self.extra_mounts.copy()

        # Add cargo cache mounts
        home = Path.home()
        cargo_registry = home / ".cargo/registry"
        cargo_git = home / ".cargo/git"

        # Check if they exist on host before mounting
        if cargo_registry.exists():
            mounts[str(cargo_registry)] = "/usr/local/cargo/registry"
        if cargo_git.exists():
            mounts[str(cargo_git)] = "/usr/local/cargo/git"

        # Add sccache mount
        sccache_dir = Path("/var/tmp/sccache")
        try:
            sccache_dir.mkdir(parents=True, exist_ok=True)
            sccache_dir.chmod(0o777)
        except Exception as e:
            logger.warning(f"Could not create/chmod sccache dir {sccache_dir}: {e}")

        if sccache_dir.exists():
            mounts[str(sccache_dir)] = "/var/tmp/sccache"

        # 3. Initialize and start workspace
        machine = platform.machine().lower()
        container_platform = (
            "linux/arm64" if "arm" in machine or "aarch64" in machine else "linux/amd64"
        )

        extra_env = {}
        enable_host_gateway = False
        if self.vllm_port is not None:
            extra_env["VLLM_HOST"] = "host.docker.internal"
            extra_env["VLLM_PORT"] = str(self.vllm_port)
            enable_host_gateway = True

        self._workspace = MountableDockerWorkspace(
            server_image=self.image,
            platform=container_platform,
            mount_dir=str(self._temp_repo.local_dir),
            extra_mounts=mounts,
            forward_env=self.forward_env,
            extra_env=extra_env,
            enable_host_gateway=enable_host_gateway,
        )
        self._exit_stack.enter_context(self._workspace)

        return self

    def push_exam(self, message: str) -> str | None:
        """Commit changes, push to original project, and return commit hash."""
        logger.info(f"Pushing coding exam commit: {message} ({self.branch_name})")
        # 1. Check for changes
        self.cloned_repo.add()
        status = self.cloned_repo.run_git(["status", "--porcelain"])
        if not status:
            logger.warning("No changes detected in the repository")
            # If no changes, still return the current commit hash
            return None

        # 2. Commit changes
        logger.debug(f"Committing changes: {message}")
        self.cloned_repo.commit(message)

        # 3. Push branch to origin
        logger.debug(f"Pushing branch {self.branch_name} to origin")
        self.cloned_repo.push("origin", self.branch_name)

        commit_hash = self.cloned_repo.rev_parse()
        logger.success(f"Commit pushed: {commit_hash}")
        return commit_hash

    def run_test(self) -> TestResult:
        try:
            process = subprocess.run(
                ["cargo", "test"],
                cwd=self.cloned_repo.local_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            return TestResult(
                is_success=process.returncode == 0,
                stdout=process.stdout,
                stderr=process.stderr,
            )
        except Exception as e:
            msg = f"Cargo test execution failed: {e}"
            logger.error(msg)
            raise TemporalCodingRepositoryError(msg) from e

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    @property
    def workspace(self) -> MountableDockerWorkspace:
        if self._workspace is None:
            raise TemporalCodingRepositoryError("Workspace not initialized")
        return self._workspace
