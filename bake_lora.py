from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

type PathLike = str | Path


def bake_lora(base_model_name: PathLike, lora_path: PathLike, output_path: PathLike):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    lora_model: PeftModel = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = lora_model.merge_and_unload()  # type: ignore
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    base_model_name = "Qwen/Qwen3-4B"
    lora_path = Path("checkpoints/qwen3-4b-sqlglot-finetuned-2025-12-23")
    output_path = Path("./checkpoints/merged/qwen3-4b-sqlglot-merged-conversational")
    bake_lora(base_model_name, lora_path, output_path)
