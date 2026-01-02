from adapter.exam.repository import GitRepositoryDict
from typing import TypedDict
import polars as pl
from pathlib import Path
from pydantic import BaseModel

from adapter.exam.repository import GitRepository

from loguru import logger


class CodingExam(BaseModel):
    id: str
    image_name: str
    project: GitRepository
    library: GitRepository
    solution_commit: str
    problem_commit: str
    question: str


class CodingExamDict(TypedDict):
    id: str
    image_name: str
    project: GitRepositoryDict
    library: GitRepositoryDict
    solution_commit: str
    problem_commit: str
    question: str


def load_exam_from_csv(
    csv_path: Path, exam_id: str, image_name: str, project_dir: Path, library_dir: Path
) -> CodingExam:
    df = pl.read_csv(csv_path)
    exam_row = df.filter(pl.col("id") == exam_id)

    if exam_row.is_empty():
        raise ValueError(f"Exam ID {exam_id} not found in {csv_path}")

    row = exam_row.to_dict(as_series=False)
    # Extract values (first item of the list for each column)
    logger.debug(f"Row data: {row}")
    return CodingExam(
        id=row["id"][0],
        image_name=row.get("image_name", [image_name])[0],
        project=GitRepository(
            name="rust-benchmarks", local_dir=project_dir
        ),  # Reconstruct assuming same project
        library=GitRepository(
            name=library_dir.name, local_dir=library_dir
        ),  # Reconstruct assuming same library
        # Note: In a real scenario you might need to infer repo names from CSV if they vary
        solution_commit=row["solution_commit"][0],
        problem_commit=row["problem_commit"][0],
        question=row["question"][0],
    )


def load_exams(
    csv_path: Path, image_name: str, project_dir: Path, library_dir: Path
) -> list[CodingExam]:
    df = pl.read_csv(csv_path)
    exams = []
    for row in df.iter_rows(named=True):
        exams.append(
            CodingExam(
                id=row["id"],
                image_name=row.get("image_name") or image_name,
                project=GitRepository(name="rust-benchmarks", local_dir=project_dir),
                library=GitRepository(name=library_dir.name, local_dir=library_dir),
                solution_commit=row["solution_commit"],
                problem_commit=row["problem_commit"],
                question=row["question"],
            )
        )
    return exams
