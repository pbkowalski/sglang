"""
Run a small model with a couple prompts and optional profiling.

Examples:
python3 scripts/profile_small_run.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4
python3 scripts/profile_small_run.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --profile --profile-by-stage
python3 scripts/profile_small_run.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --profile --merge-profiles --profile-annotate-iterations --profile-classify-kernels
"""

import argparse
import multiprocessing
import time

import requests

from sglang.profiler import run_profile
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

DEFAULT_TIMEOUT = 600
DEFAULT_PROMPTS = [
    "Explain CUDA streams in one paragraph.",
    "Write a short haiku about profiling.",
]


def _launch_server_internal(server_args: ServerArgs):
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(multiprocessing.current_process().pid, include_parent=False)


def _launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=_launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"

    start_time = time.time()
    while time.time() - start_time < DEFAULT_TIMEOUT:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("Server failed to start within the timeout period.")


def _send_prompts(base_url: str, prompts, max_new_tokens: int, temperature: float):
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": False,
    }
    response = requests.post(
        base_url + "/generate",
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text (can be passed multiple times).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for a few forward steps.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=5,
        help="Number of steps to profile.",
    )
    parser.add_argument(
        "--profile-by-stage",
        action="store_true",
        help="Profile prefill and decode separately.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help="Directory for profiler traces.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default=None,
        help="Prefix for profiler trace filenames.",
    )
    parser.add_argument(
        "--merge-profiles",
        action="store_true",
        help="Merge profiles across ranks after profiling.",
    )

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    if not server_args.model_path:
        server_args.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"

    prompts = args.prompt or DEFAULT_PROMPTS

    proc = None
    try:
        proc, base_url = _launch_server_process(server_args)

        if args.profile:
            run_profile(
                url=base_url,
                num_steps=args.profile_steps,
                activities=["CPU", "GPU"],
                output_dir=args.profile_output_dir,
                profile_by_stage=args.profile_by_stage,
                merge_profiles=args.merge_profiles,
                profile_prefix=args.profile_prefix,
            )

        result = _send_prompts(
            base_url=base_url,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("Generate response:")
        print(result)
    finally:
        if proc is not None:
            kill_process_tree(proc.pid, include_parent=True)


if __name__ == "__main__":
    main()
