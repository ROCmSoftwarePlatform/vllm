import time

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        '/data/AI-ModelScope/Mixtral-8x7B-Instruct-v0___1/',
        tensor_parallel_size=1,
        #quantization="serenity",
        dtype='float16',
        #swap_space=16,
        #enforce_eager=True,
        #kv_cache_dtype="fp8",
        #quantization="fp8",
        #quantized_weights_path="/quantized/quark/llama.safetensors",
        #worker_use_ray=True,
        #trust_remote_code=True,
        #distributed_executor_backend="mp",
    )
    batch_size = 5
    max_tokens = 256
    prompt = """The sun is a"""
    sampling_params = SamplingParams(temperature=0,
                                     top_p=0.95,
                                     max_tokens=max_tokens)

    start_time = time.perf_counter()
    outs = llm.generate([prompt] * batch_size, sampling_params=sampling_params)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    out_lengths = [len(x.token_ids) for out in outs for x in out.outputs]
    num_tokens = sum(out_lengths)

    print(
        f"{num_tokens} tokens. {num_tokens / batch_size} on average. {num_tokens / elapsed_time:.2f} tokens/s. {elapsed_time} seconds"  # noqa: E501
    )
    for out in outs:
        print("===========")
        print(out.outputs[0].text)


if __name__ == "__main__":
    main()
