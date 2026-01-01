from typing import override
import os
import threading
import uuid
from typing import Any

from openhands.sdk.logger import get_logger
from openhands.workspace.docker.workspace import (
    DockerWorkspace,
    check_port_available,
    execute_command,
    find_available_tcp_port,
)
from pydantic import Field

logger = get_logger(__name__)


class MountableDockerWorkspace(DockerWorkspace):
    extra_mounts: dict[str, str] = Field(
        default_factory=dict
    )  # host_dir: container_dir

    @override
    def _start_container(self, image: str, context: Any) -> None:
        """Start the Docker container with the given image.

        This method handles all container lifecycle: port allocation, Docker
        validation, container creation, health checks, and RemoteWorkspace
        initialization.

        Args:
            image: The Docker image tag to use.
            context: The Pydantic context from model_post_init.
        """
        # Store the image name for cleanup
        self._image_name = image

        # Determine port
        if self.host_port is None:
            self.host_port = find_available_tcp_port()
        else:
            self.host_port = int(self.host_port)

        if not check_port_available(self.host_port):
            raise RuntimeError(f"Port {self.host_port} is not available")

        if self.extra_ports:
            if not check_port_available(self.host_port + 1):
                raise RuntimeError(
                    f"Port {self.host_port + 1} is not available for VSCode"
                )
            if not check_port_available(self.host_port + 2):
                raise RuntimeError(
                    f"Port {self.host_port + 2} is not available for VNC"
                )

        # Ensure docker is available
        docker_ver = execute_command(["docker", "version"]).returncode
        if docker_ver != 0:
            raise RuntimeError(
                "Docker is not available. Please install and start "
                "Docker Desktop/daemon."
            )

        # Prepare Docker run flags
        flags: list[str] = []
        for key in self.forward_env:
            if key in os.environ:
                flags += ["-e", f"{key}={os.environ[key]}"]

        if self.mount_dir:
            mount_path = "/workspace"
            flags += ["-v", f"{self.mount_dir}:{mount_path}"]
            logger.info(
                "Mounting host dir %s to container path %s",
                self.mount_dir,
                mount_path,
            )

        for host_dir, container_dir in self.extra_mounts.items():
            flags += ["-v", f"{host_dir}:{container_dir}"]

        ports = ["-p", f"{self.host_port}:8000"]
        if self.extra_ports:
            ports += [
                "-p",
                f"{self.host_port + 1}:8001",  # VSCode
                "-p",
                f"{self.host_port + 2}:8002",  # Desktop VNC
            ]
        flags += ports

        # Add GPU support if enabled
        if self.enable_gpu:
            flags += ["--gpus", "all"]

        # Run container
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
            "--name",
            f"agent-server-{uuid.uuid4()}",
            *flags,
            image,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        proc = execute_command(run_cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to run docker container: {proc.stderr}")

        self._container_id = proc.stdout.strip()
        logger.info("Started container: %s", self._container_id)

        # Optionally stream logs in background
        if self.detach_logs:
            self._logs_thread = threading.Thread(
                target=self._stream_docker_logs, daemon=True
            )
            self._logs_thread.start()

        # Set host for RemoteWorkspace to use
        # The container exposes port 8000, mapped to self.host_port
        # Override parent's host initialization
        object.__setattr__(self, "host", f"http://localhost:{self.host_port}")
        object.__setattr__(self, "api_key", None)

        # Wait for container to be healthy
        self._wait_for_health()
        logger.info("Docker workspace is ready at %s", self.host)
