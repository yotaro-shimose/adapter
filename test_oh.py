import os

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace, DockerDevWorkspace
from dotenv import load_dotenv
import platform

load_dotenv()
llm = LLM(
    model=os.getenv("LLM_MODEL", "azure/gpt-5-mini"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL", None),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ],
)


def detect_platform():
    """Detects the correct Docker platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


# Use your custom Rust image
with (
    DockerDevWorkspace(
        server_image="ohserver-rust",  # Your custom image with Rust
        platform=detect_platform(),
    ) as workspace
    # DockerDevWorkspace(
    #     base_image="ohserver-rust",
    #     host_port=8000,
    #     target="source",
    # ) as workspace
):
    conversation = Conversation(agent=agent, workspace=workspace)

    # Agent creates project INSIDE the container
    conversation.send_message(
        "First check if cargo is available. And then create a Rust project with a simple add function and write tests for it."
    )
    conversation.run()

    # Validate by running tests inside container
    result = workspace.execute_command("cargo test")
    print(f"Cargo test output: {result.stdout}")
    print(f"Cargo test success: {result.exit_code == 0}")

print("All done!")
