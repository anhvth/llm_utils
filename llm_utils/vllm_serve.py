from glob import glob
import os
import subprocess
import time
from typing import List, Optional
from fastcore.script import call_parse
from loguru import logger
import argparse
import requests


LORA_DIR = os.environ.get("LORA_DIR", "/loras")
LORA_DIR = os.path.abspath(LORA_DIR)
logger.info(f"LORA_DIR: {LORA_DIR}")

def kill_existing_vllm(vllm_binary: Optional[str] = None) -> None:
    """Kill selected vLLM processes using fzf."""
    if not vllm_binary:
        vllm_binary = get_vllm()

    # List running vLLM processes
    result = subprocess.run(
        f"ps aux | grep {vllm_binary} | grep -v grep",
        shell=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip().split("\n")

    if not processes or processes == [""]:
        print("No running vLLM processes found.")
        return

    # Use fzf to select processes to kill
    fzf = subprocess.Popen(
        ["fzf", "--multi"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    selected, _ = fzf.communicate("\n".join(processes))

    if not selected:
        print("No processes selected.")
        return

    # Extract PIDs and kill selected processes
    pids = [line.split()[1] for line in selected.strip().split("\n")]
    for pid in pids:
        subprocess.run(
            f"kill -9 {pid}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Killed processes: {', '.join(pids)}")


def add_lora(
    lora_name_or_path: str,
    url: str = "http://HOST:PORT/v1/load_lora_adapter",
    host_port: str = "localhost:8150",
    served_model_name:str = None
) -> dict:
    if os.path.exists(lora_name_or_path):
        assert served_model_name, "served_model_name is required when lora_name_or_path is a path"
        # should be located at LORA_DIR/{LORA_NAME}
        lora_name_or_path = os.path.abspath(lora_name_or_path)
        if not lora_name_or_path.startswith(LORA_DIR):
            # copy to LORA_DIR
            paths = glob(f"{lora_name_or_path}/*")
            import shutil
            target_dir = os.path.join(LORA_DIR, served_model_name)
            # if os.path.isdir(target_dir):
            #     shutil.rmtree(target_dir)
            # do not copy the
            #copy everything except folder
            os.makedirs(target_dir, exist_ok=True)
            for path in paths:
                if os.path.isfile(path):
                    try:
                        print(f"Copying {path} to {target_dir}")
                        shutil.copy(path, target_dir)
                    except:
                        pass
            # shutil.copytree(lora_name_or_path, target_dir)
    else:
        served_model_name = lora_name_or_path.split(LORA_DIR)[1].strip("/")
        
    logger.info(f'LOra name: {lora_name_or_path}')
    url = url.replace("HOST:PORT", host_port)
    if not lora_name_or_path.startswith(LORA_DIR):
        lora_path = os.path.join(LORA_DIR, served_model_name)
    else:
        lora_path = lora_name_or_path
    logger.info(f'{url=}')
    try:
        unload_lora(lora_name_or_path, host_port=host_port)
    except Exception as e:
        pass
    headers = {"Content-Type": "application/json"}
    lora_name_or_path = served_model_name or lora_name_or_path
    data = {"lora_name": lora_name_or_path, "lora_path": lora_path}
    logger.info(f"{data=}")
    # logger.warning(f"Failed to unload LoRA adapter: {str(e)}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Handle potential non-JSON responses
        try:
            return response.json()
        except ValueError:
            return {
                "status": "success",
                "message": (
                    response.text
                    if response.text.strip()
                    else "Request completed with empty response"
                ),
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}


def unload_lora(lora_name, host_port):
    try:
        url = f"http://{host_port}/v1/unload_lora_adapter"
        logger.info(f'{url=}')
        headers = {"Content-Type": "application/json"}
        data = {"lora_name": lora_name}
        logger.info(f"Unloading LoRA adapter: {data=}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.success(f"Unloaded LoRA adapter: {lora_name}")
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}



def serve(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str] = None,
    port_start: int = 8155,
    gpu_memory_utilization: float = 0.93,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    enable_lora: bool = False,
    is_bnb: bool = False,
    not_verbose=True,
    extra_args: Optional[List[str]] = [],
):
    """Main function to start or kill vLLM containers."""

    """Start vLLM containers with dynamic args."""
    print("Starting vLLM containers...,")
    gpu_groups_arr = gpu_groups.split(",")
    VLLM_BINARY = get_vllm()
    if enable_lora:
        VLLM_BINARY = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True " + VLLM_BINARY

    # Auto-detect quantization based on model name if not explicitly set
    if not is_bnb and model and ("bnb" in model.lower() or "4bit" in model.lower()):
        is_bnb = True
        print(f"Auto-detected quantization for model: {model}")

    # Set environment variables for LoRA if needed
    if enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        print("Enabled runtime LoRA updating")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = port_start + i
        gpu_group = ",".join([str(x) for x in gpu_group])
        tensor_parallel = len(gpu_group.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_group}",
            VLLM_BINARY,
            "serve",
            model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
            "--disable-log-requests",
            "--enable-prefix-caching",
        ]
        if not_verbose or True:
            cmd += ["--uvicorn-log-level critical"]

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        if is_bnb:
            cmd.extend(
                ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
            )

        if enable_lora:
            cmd.extend(["--fully-sharded-loras", "--enable-lora"])
        # add kwargs
        if extra_args:
            for name_param in extra_args:
                name, param = name_param.split("=")
                cmd.extend([f"{name}", param])
        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = (
            f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"
        )

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)


def get_vllm():
    VLLM_BINARY = subprocess.check_output("which vllm", shell=True, text=True).strip()
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)
    logger.info(f"vLLM binary: {VLLM_BINARY}")
    assert os.path.exists(
        VLLM_BINARY
    ), f"vLLM binary not found at {VLLM_BINARY}, please set VLLM_BINARY env variable"
    return VLLM_BINARY


def get_args():
    """Parse command line arguments."""
    example_args = [
        "svllm add_lora --model LOra_name --host localhost:8150",
        "svllm add_lora lora_name@path:port",
        "svllm kill",
    ]

    parser = argparse.ArgumentParser(
        description="vLLM Serve Script", epilog="Example: " + " || ".join(example_args)
    )
    parser.add_argument(
        "mode", choices=["serve", "kill", "add_lora", "unload_lora"], help="Mode to run the script in"
    )
    parser.add_argument("--model", "-m", type=str, help="Model to serve")
    parser.add_argument(
        "--gpu_groups", "-g", type=str, help="Comma-separated list of GPU groups"
    )
    parser.add_argument(
        "--served_model_name", type=str, help="Name of the served model"
    )
    parser.add_argument(
        "--port_start", type=int, default=8155, help="Starting port number"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument(
        "--max_model_len", type=int, default=8192, help="Maximum model length"
    )
    # parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA")
    parser.add_argument(
        "--disable_lora",
        dest="enable_lora",
        action="store_false",
        help="Enable LoRA",
        default=True,
    )
    parser.add_argument("--bnb", action="store_true", help="Enable quantization")
    parser.add_argument(
        "--not_verbose", action="store_true", help="Disable verbose logging"
    )
    parser.add_argument("--vllm_binary", type=str, help="Path to the vLLM binary")
    parser.add_argument("--lora_name", type=str, help="Name of the LoRA adapter")
    parser.add_argument(
        "--pipeline-parallel",
        "-pp",
        default=1,
        type=int,
        help="Number of pipeline parallel stages",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for the serve command",
    )
    parser.add_argument(
        "--host_port", type=str, default="HOST:PORT", help="Host to serve the model"
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""

    args = get_args()
    # if help
    if args.mode == "serve":
        serve(
            args.model,
            args.gpu_groups,
            args.served_model_name,
            args.port_start,
            args.gpu_memory_utilization,
            args.dtype,
            args.max_model_len,
            args.enable_lora,
            args.bnb,
            args.not_verbose,
            args.extra_args,
        )
    elif args.mode == "kill":
        kill_existing_vllm(args.vllm_binary)
    elif args.mode == "add_lora":
        lora_name = args.model
        add_lora(lora_name, host_port=args.host_port, served_model_name=args.served_model_name)
    elif args.mode == "unload_lora":
        lora_name = args.model
        unload_lora(lora_name, host_port=args.host_port)

if __name__ == "__main__":
    main()
