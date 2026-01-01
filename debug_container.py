import time
import signal
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from adapter.exam.renv import RustCodingEnvironment
from loguru import logger
import polars as pl

# Import from the sibling script
from create_coding_exam import (
    Config,
    CodingExam,
    GitRepository,
    gen_id,
)


def handle_interrupt(sig, frame):
    logger.info("\nInterrupted by user. Cleaning up...")
    sys.exit(0)


def launch_debug_container(config: Config, exam: CodingExam | None = None):
    """
    Launch a temporal docker container for manual debugging.
    The container will have the project and library mounted/cloned.
    If 'exam' is provided, it sets up the specific exam environment (repos and commit).
    """

    if exam:
        project_repo = exam.project
        library_repo = exam.library
        logger.info(f"Using exam configuration for ID: {exam.id}")
        logger.info(f"Target commit: {exam.problem_commit}")
    else:
        # Defaults if no exam provided
        project_repo = GitRepository(
            name="rust-benchmarks",
            local_dir=config.project_dir,
        )
        library_repo = GitRepository(
            name=config.library_dir.name, local_dir=config.library_dir
        )

    branch_name = gen_id("debug")
    logger.info(f"Starting debug session for library: {library_repo.name}")

    try:
        # Use RustCodingEnvironment for unified setup
        with RustCodingEnvironment(
            branch_name=branch_name,
            project=project_repo,
            library=library_repo,
            image=config.image_name,
        ) as env:
            logger.info(f"Rust environment set up at: {env.cloned_repo.local_dir}")

            if exam:
                logger.info(f"Checking out solution_commit: {exam.solution_commit}")
                env.cloned_repo.checkout(exam.solution_commit)

            workspace = env.workspace
            container_id = workspace._container_id
            logger.success(f"Container launched! ID: {container_id}")
            logger.info("Project is mounted at /workspace")
            logger.info("You can attach to the container using:")
            logger.info(f"  docker exec -it {container_id} /bin/bash")

            logger.info("Press Ctrl+C to stop the container and cleanup.")

            # Keep alive loop
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt caught. Cleaning up...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def load_exam_from_csv(csv_path: Path, exam_id: str, config: Config) -> CodingExam:
    df = pl.read_csv(csv_path)
    exam_row = df.filter(pl.col("id") == exam_id)

    if exam_row.is_empty():
        raise ValueError(f"Exam ID {exam_id} not found in {csv_path}")

    row = exam_row.to_dict(as_series=False)
    # Extract values (first item of the list for each column)
    return CodingExam(
        id=row["id"][0],
        image_name=row.get("image_name", [config.image_name])[
            0
        ],  # Use image_name from CSV or fallback to config
        project=GitRepository(
            name="rust-benchmarks", local_dir=config.project_dir
        ),  # Reconstruct assuming same project
        library=GitRepository(
            name=config.library_dir.name, local_dir=config.library_dir
        ),  # Reconstruct assuming same library
        # Note: In a real scenario you might need to infer repo names from CSV if they vary
        solution_commit=row["solution_commit"][0],
        problem_commit=row["problem_commit"][0],
        question=row["question"][0],
    )


def main():
    load_dotenv()
    config = Config.default()

    parser = argparse.ArgumentParser(
        description="Launch a debug container for OpenHands"
    )
    parser.add_argument("--exam-id", type=str, help="ID of the exam to debug")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_interrupt)

    exam = None
    if args.exam_id:
        try:
            logger.info(f"Loading exam {args.exam_id} from {config.output_file}...")
            exam = load_exam_from_csv(config.output_file, args.exam_id, config)
        except Exception as e:
            logger.error(f"Failed to load exam: {e}")
            sys.exit(1)
    else:
        # Randomly sample an exam if no ID provided
        logger.info(
            f"No exam ID provided. Sampling a random exam from {config.output_file}..."
        )
        try:
            df = pl.read_csv(config.output_file)
            df = df.filter(pl.col("status") == "generated")
            if not df.is_empty():
                random_exam_row = df.sample(1)
                random_id = random_exam_row["id"][0]
                logger.info(f"Randomly selected exam ID: {random_id}")
                exam = load_exam_from_csv(config.output_file, random_id, config)
            else:
                logger.warning(
                    "Exam file is empty or not found. Proceeding with default environment."
                )
        except Exception as e:
            logger.error(
                f"Failed to sample random exam: {e}. Proceeding with default environment."
            )

    try:
        launch_debug_container(config, exam)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
