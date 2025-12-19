from textwrap import dedent

from adapter.utils.savable import Savable


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
