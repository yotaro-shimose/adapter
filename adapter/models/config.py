from pathlib import Path
from oai_utils.agent import AgentsSDKModel
from dataclasses import dataclass


@dataclass
class ProblemCreationConfig:
    repo_path: Path
    topic_extraction_semaphore: int
    question_generation_semaphore: int
    max_topics: int
    batch_size: int
    output_dir: Path
    model: AgentsSDKModel

    @property
    def repo_name(self) -> str:
        return self.repo_path.name

    @property
    def repo_output_dir(self) -> Path:
        path = self.output_dir / self.repo_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def topic_save_path(self) -> Path:
        return self.repo_output_dir / "topics.json"

    @property
    def output_path(self) -> Path:
        return self.repo_output_dir / "problems.json"
