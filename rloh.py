import asyncio
from pathlib import Path
from typing import Any, cast

import agentlightning as agl
from agentlightning.types import AttemptedRollout
from dotenv import load_dotenv
from loguru import logger
import os
from adapter.exam.exam import CodingExam, CodingExamDict, load_exams
from adapter.exam.solver import Solver
from create_coding_exam import ExamConfig

project_name = "RLOH"
experiment_name = "Test"
RL_TRAINING_CONFIG: dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,
        "max_prompt_length": 30000,
        "max_response_length": 10000,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 1,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "qwen3_coder",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 8,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": False,
                "optimizer_offload": False,
            },
            "checkpoint": {"save_contents": ["hf_model"]},
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": False},
        },
        "model": {
            "path": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "nnodes": 1,
        "n_gpus_per_node": 8,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": project_name,
        "experiment_name": experiment_name,
        "test_freq": 32,
        "total_epochs": 2,
        "save_freq": 32,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
        "default_local_dir": f"./checkpoints/{project_name}/{experiment_name}",
    },
}


class LitOHAgent(agl.LitAgent):
    async def rollout_async(
        self,
        task: CodingExamDict,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        exam = CodingExam.model_validate(task)
        agl_llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        base_url = agl_llm.get_base_url(
            rollout.rollout_id, cast(AttemptedRollout, rollout).attempt.attempt_id
        )

        logger.debug("*" * 50)
        logger.debug(f"Base URL: {base_url}")
        logger.debug("*" * 50)

        solver = Solver.create(
            model=f"hosted_vllm/{agl_llm.model}",
            base_url=base_url,
            api_key=agl_llm.api_key,
        )
        value = await asyncio.to_thread(solver.solve_exam, exam)
        return float(value)


def train(config: dict[str, Any]) -> None:
    """Train the SQL agent with the given configuration."""
    csv_path = Path("exams.csv")
    exam_config = ExamConfig.default()

    agent = LitOHAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=1, algorithm=algorithm)
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore
    exams = [
        exam.model_dump(mode="json")
        for exam in load_exams(
            csv_path=csv_path,
            image_name=exam_config.image_name,
            project_dir=exam_config.project_dir,
            library_dir=exam_config.library_dir,
        )
    ]
    trainer.fit(agent, train_dataset=exams, val_dataset=exams)


def main() -> None:
    cfg_type = "qwen"
    # Get the appropriate configuration

    print(f"Starting training with '{cfg_type}' configuration...")
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    train(RL_TRAINING_CONFIG)


if __name__ == "__main__":
    load_dotenv()
    main()
