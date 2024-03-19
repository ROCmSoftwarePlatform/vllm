#!/usr/bin/env python3

#
#   This is a quick hack that produces PPL measurement by 
#   iteratively dumping the logprob vector for the single next symbol 
#   that is to be generated over the preloaded context.
#   It is actually an *inefficient* procedure because for the
#   N-token string it takes N*(preload + generation) time instead of
#   preload + N*generation
#

import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import torch
import sys
import datetime
import json
import argparse
from vllm import LLM, SamplingParams
import math
import operator

def get_wikitext2_text(tokenizer):
    with open(args.data) as f:
        test_text = "\n".join(line.strip() for line in f)
        test_enc = tokenizer(test_text)

    return test_enc, test_text

def vllm_init(args):
    llm = LLM(
        model=args.model,
        tokenizer=None,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_scales_path=args.kv_cache_scales_path if args.kv_cache_scales_path!='' else None,
        max_context_len_to_capture=args.context_size,
    )

    sampling_params = SamplingParams(
        n=1, 
        temperature=0.0, 
        top_p=1,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=1,
        logprobs=32000,
        prompt_logprobs=32000,
        presence_penalty=0.0
    )

    return llm, sampling_params

def vllm_predict(CONT, llm, sampl_par):
    result=llm.generate( prompt_token_ids=CONT,sampling_params=sampl_par)
    return result

def main(args: argparse.Namespace):

    print (f"### Initialising @ {datetime.datetime.now()}")
    my_ppl=0.0

    my_tokenizer = LlamaTokenizer.from_pretrained(args.model)
    print("Loaded the tokenizer.")

    print("*** Initializing the engine.")
    my_llm, my_sampl_par = vllm_init(args)
    print(my_sampl_par)
    print("*** Initialized the engine.")

    my_test_enc, my_test_text = get_wikitext2_text(my_tokenizer)
    print("Loaded the test data.") 

    my_n_samples = (len(my_test_enc['input_ids'])-1)//(args.batch_size*args.context_size)

    print (f"### Starting generation @ {datetime.datetime.now()}")
    for c in range(my_n_samples):
        CONTEXT = []
        for r in range(args.batch_size):
            CONTEXT.append(my_test_enc['input_ids'][(c*args.batch_size+r)*args.context_size:(c*args.batch_size+r+1)*args.context_size])

        LOGPROBS = vllm_predict(CONTEXT, my_llm, my_sampl_par)
    
        for r in range(args.batch_size):
            start=args.context_size//2
            my_stat_size=args.context_size-start+1 
            for pc in range(start,args.context_size): 
                my_ppl -= LOGPROBS[r].prompt_logprobs[pc][my_test_enc['input_ids'][(c*args.batch_size+r)*args.context_size+pc]]
            my_ppl -= LOGPROBS[r].outputs[0].logprobs[0][my_test_enc['input_ids'][(c*args.batch_size+r+1)*args.context_size]]
        print(f"Intermediate estimates:\n\tCross-entropy_intermediate={my_ppl/((c+1)*args.batch_size*my_stat_size)}")

    my_ppl/=(my_n_samples*args.batch_size*my_stat_size)

    print (f"### Done @ {datetime.datetime.now()}")

    print(f"PPL={math.exp(my_ppl)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--data', type=str, default='./wikitext/wikitext-2-v1/test-00000-of-00001.parquet')
    parser.add_argument('--context-size', type=int, default=4096)
    parser.add_argument('--kv-cache-scales-path', type=str, default='')
    parser.add_argument('--num-samples-per-task', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--experiment-prefix',type=str, default='solution_samples')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--ppl-measurement', action='store_false')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8_e5m2','fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    args = parser.parse_args()

    main(args)
