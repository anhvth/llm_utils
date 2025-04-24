import os
import subprocess
import re
import threading
import time
from typing import Optional, List, Union
from pydantic import BaseModel
import json as jdumps
from .lm import OAI_LM
import socket
from loguru import logger

timeout = 300


def check_is_port_available(port: int) -> bool:
    """Check if a TCP port on localhost is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def init_server(
    model_path: str,
    port: int = 8150,
    tensor_parallel: int = 4,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    lora_path: str = None,
    host: str = "localhost",
    gpu_devices: Optional[Union[List[int], str]] = None,
    hf_home: Optional[str] = "/data-4090/huggingace-cache",
    vllm_bin: str = "/home/anhvth5/python-venv/default/bin/vllm",
    timeout: int = 300,
    **kwargs,
):
    """
    Initialize a vLLM server with the given parameters.

    Args:
        model_path: Path to the model to serve
        port: Port to serve the model on
        tensor_parallel: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: Memory utilization per GPU
        max_model_len: Maximum sequence length
        lora_modules: LoRA modules to enable, format: "name=path"
        host: Host to serve on
        gpu_devices: List of GPU devices to use or comma-separated string (e.g., "4,5,6,7")
        hf_home: Path to Hugging Face cache
        vllm_bin: Path to vLLM binary
        timeout: Timeout in seconds to wait for the server to start
    """
    # Check if port is available before starting server
    if not check_is_port_available(port):
        print(f"Port {port} is not available. Assuming server is already running.")
        return None

    env = {}
    if hf_home:
        env["HF_HOME"] = hf_home
    assert env['HF_TOKEN'] != ""
    if gpu_devices:
        if isinstance(gpu_devices, list):
            gpu_devices = ",".join(map(str, gpu_devices))
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices

    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    # Base command
    cmd = [
        vllm_bin,
        "serve",
        model_path,
        "--port",
        str(port),
        "--tensor-parallel",
        str(tensor_parallel),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        "auto",
        "--max-model-len",
        str(max_model_len),
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--uvicorn-log-level",
        "critical",
        "--enforce-eager",
    ]

    # Add LoRA-specific parameters if lora_modules is provided
    if lora_path:
        assert os.path.exists(lora_path), f"LoRA path {lora_path} does not exist."
        adapter_json_path = os.path.join(lora_path, "adapter_config.json")
        assert os.path.exists(
            adapter_json_path
        ), f"LoRA adapter config file {adapter_json_path} does not exist."
        cmd.extend(["--fully-sharded-loras", "--enable-lora"])
        cmd.extend(["--lora-modules", f"lora_model={lora_path}"])

    # Add any additional kwargs as command line args
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # Write the command to /tmp/start_vllm.sh instead of running it
    env_str = " ".join([f"{k}={v}" for k, v in env.items()])
    cmd_str = " ".join(cmd)
    full_cmd = f"{env_str} {cmd_str}"
    # wrap the full cmd to run in a tmux name vlll_{port}
    full_cmd = f"tmux new-session -d -s vllm_{port} '{full_cmd}'"
    script_path = "/tmp/start_vllm.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(full_cmd + "\n")
    os.chmod(script_path, 0o755)
    logger.info(f"vLLM server start command written to {script_path}. Please run it manually in a shell.")


class ClassificationLM(OAI_LM):
    '''
    from llm_utils.lm_classification import ClassificationLM


    evaluator_system_prompt_for_training_classification = """
    You are Translation-Evaluator-LM.

    Given a JSON object with these keys:
    - text_to_translate: the source text (Chinese)
    - translation: the candidate translation (English)
    - glossary: a mapping of required terms
    - category: one of "meaning", "structure", or "terminology"

    Your task:
    1. Judge the translation ONLY for the specified category (score 1–5, 5=best):
        - meaning: accuracy & completeness
        - structure: tags/placeholders preserved
        - terminology: glossary fidelity
    2. Output a single integer (1–5) as your answer, with no explanation or extra text.

    Example input:
    {"text_to_translate": "...", "translation": "...", "glossary": "...", "category": "structure"}

    Example output:
    4
    """



    cls_agent = ClassificationLM(
        model="TranslationEvalGeminiProDistil32B",
        ports=[8150],
        host='localhost',
        system_prompt=evaluator_system_prompt_for_training_classification,
    )
    '''

    def __init__(
        self,
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        cache: bool = True,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        host="localhost",
        port=None,
        ports=None,
        api_key=None,
        input_model: Optional[type[BaseModel]] = None,
        post_processor_to_score: bool = False,
        model_path: str = None,
        tensor_parallel: int = 4,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,
        lora_path: Optional[str] = None,
        gpu_devices: Optional[Union[List[int], str]] = None,
        hf_home: Optional[str] = "/data-4090/huggingace-cache",
        vllm_bin: str = "/home/anhvth5/python-venv/default/bin/vllm",
        start_server: bool = False,
        **kwargs,
    ):
        if start_server:
            logger.info(
                f"Starting vLLM server for model {model} on port {port} with tensor parallelism {tensor_parallel}"
            )
            init_server(
                model_path=model_path or model,
                port=port,
                tensor_parallel=tensor_parallel,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                lora_path=lora_path,
                host=host,
                gpu_devices=gpu_devices,
                hf_home=hf_home,
                vllm_bin=vllm_bin,
                timeout=timeout,
                **kwargs.get("server_kwargs", {}),
            )
        if lora_path:
            logger.info(f"Loading LoRA model from {lora_path} Use `lora_model` as model name")
            model = "lora_model"

        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            host=host,
            port=port,
            ports=ports or ([port] if port else None),
            api_key=api_key or "abc",
            **kwargs,
        )
        self.input_model = input_model
        self.system_prompt = system_prompt
        self.post_processor_to_score = post_processor_to_score

    def get_messages(self, input: Union[str, dict]):

        if isinstance(input, str):
            user_msg = input
        elif not isinstance(input, dict):
            # load it to input model
            input_obj = self.input_model(**input)
            user_msg = jdumps(input_obj.model_dump())
        else:
            raise ValueError("input must be str or dict")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return messages

    def __call__(self, input, **kwargs):
        messages = self.get_messages(input)
        output = super().__call__(messages=messages, **kwargs)
        return output

    # del process to kill thread
    def __del__(self):
        if hasattr(self, "process") and self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("vLLM server process terminated.")
        else:
            logger.info("No vLLM server process to terminate.")