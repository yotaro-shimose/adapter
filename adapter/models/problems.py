from adapter.solver.constant import SOLVER_PROMPT
from textwrap import dedent
from typing import Self
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
            completion.append(
                f"<think>\n{sample.reasoning}\n</think>\n\n{sample.answer}"
            )
        dataframe = pl.DataFrame({"prompt": prompt, "completion": completion})
        return Dataset.from_polars(dataframe)

    def as_conversational(self, system_prompt: str = SOLVER_PROMPT) -> Dataset:
        items = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": sample.question,
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>{sample.reasoning}</think>{sample.answer}",
                    },
                ]
            }
            for sample in self.problems
        ]
        return Dataset.from_list(items)

    def sort(self) -> Self:
        return self.__class__(problems=sorted(self.problems, key=lambda x: x.question))

    def head(self, n: int) -> Self:
        return self.__class__(problems=self.problems[:n])
