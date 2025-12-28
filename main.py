from adapter.verifier.verify import verify_problem
from pathlib import Path

from swerex.deployment.docker import DockerDeployment

from adapter.env import ProgrammingEnvironment
from adapter.solver import ProblemSolver


async def main():
    doc_path = Path("json_schema.md")
    document = doc_path.read_text()

    # problem = await create_question(document)
    problem = ...
    print(problem.as_md())
    print("Verifying problem...")
    deployment = DockerDeployment(image="sandbox-ver0")
    is_correct = await verify_problem(problem, deployment)
    if not is_correct:
        print("Problem verification failed.")

    new_deployment = DockerDeployment(image="sandbox-ver0")

    solver = ProblemSolver.create()
    env = await ProgrammingEnvironment.create(deployment=new_deployment)
    await solver.solve(problem, env)
    print("Solution deployment ready.")

    print("Verifying solution...")
    is_solution_correct = await env.is_passing(problem)
    if is_solution_correct:
        print("Solution is correct!")
    else:
        print("Solution is incorrect.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
