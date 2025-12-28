import time

from oai_utils.vllm import VLLMSetup


async def main():
    data_parallel_size = 1
    vllm_setup = VLLMSetup.qwen3(data_parallel_size=data_parallel_size)
    # vllm_setup = VLLMSetup(
    #     model="qwen3-4b-sqlglot-merged",
    #     reasoning_parser="deepseek_r1",
    #     data_parallel_size=data_parallel_size,
    # )
    await vllm_setup.ensure_vllm_running()
    while True:
        time.sleep(100)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
