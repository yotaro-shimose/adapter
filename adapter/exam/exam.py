from pydantic import BaseModel

from adapter.exam.repository import GitRepository


class CodingExam(BaseModel):
    id: str
    image_name: str
    project: GitRepository
    library: GitRepository
    solution_commit: str
    problem_commit: str
    question: str
