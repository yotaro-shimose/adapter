from textwrap import dedent

from adapter.utils.savable import Savable
from datasets import Dataset
import polars as pl


class VerifiableProblem(Savable):
    task_name: str
    problem_statement: str
    canonical_solution: str
    test_code: str

    def as_md(self) -> str:
        return dedent(
            f"""\
        ## {self.task_name}
        ### Problem Statement
        {self.problem_statement}
        ### Canonical Solution
        ```python
        {self.canonical_solution}
        ```
        ### Test Code
        ```python
        {self.test_code}
        ```
        """
        )


class QAProblem(Savable):
    question: str
    answer: str


class ProblemDataset(Savable):
    problems: list[VerifiableProblem | QAProblem]


class QRA(Savable):
    question: str
    answer: str
    reasoning: str


class QRADataset(Savable):
    problems: list[QRA]

    def as_prompt_completion(self) -> Dataset:
        prompt = []
        completion = []
        for sample in self.problems:
            prompt.append(sample.question)
            completion.append(f"<think>{sample.reasoning}</think>{sample.answer}")
        dataframe = pl.DataFrame({"prompt": prompt, "completion": completion})
        return Dataset.from_polars(dataframe)
