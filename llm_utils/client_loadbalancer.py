import asyncio
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union, Dict

import subprocess
import logging
import requests

from loguru import logger
from openai import OpenAI  # Ensure you have the OpenAI package installed
from speedy_utils import dump_json_or_pickle, identify_uuid, load_by_ext  # Ensure you have this utility or implement similar functions
from tqdm.asyncio import tqdm_asyncio

# Define cache directory
CACHE_DIR = str(Path("~/.cache/llm_cache").expanduser().resolve())
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure Loguru logger


def _start_server(
    gpu_ids: List[int],
    model_path: str,
    base_port: int,
    verbose: bool = False,
    use_docker: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "half",
    enforce_eager: bool = True,
    max_model_len: int = 2048,
    swap_space: int = 4,
    vllm_path: str = "/home/anhvth5/miniconda3/envs/py312-vllm/bin/vllm",
    additional_volumes: Optional[List[str]] = None
):

    model_path = os.path.abspath(model_path)
    additional_volumes = additional_volumes or []

    tensor_parallel_size = len(gpu_ids)
    gpu_ids_str = ','.join(map(str, gpu_ids))


    # Construct the command to run directly on the host
    host_command = [
        f"CUDA_VISIBLE_DEVICES={gpu_ids_str}",
        vllm_path, "serve",
        os.path.abspath(model_path),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", dtype,
        "--enforce-eager" if enforce_eager else "--no-enforce-eager",
        "--max-model-len", str(max_model_len),
        "--swap-space", str(swap_space),
        "--port", str(base_port)
    ]

    # Join the command into a single string for tmux
    cmd_str = ' '.join(host_command)
    # Wrap the command in a tmux session
    command = f'tmux new-session -d -s vllm_{base_port} "{cmd_str}"'
    shell = True

    if verbose:
        # Print the command for debugging purposes
        if use_docker:
            print("Executing Docker command:", ' '.join(command))
        else:
            print("Executing Host command:", command)
    else:
        # Log the start of the server
        logger.info(f"Starting server for GPUs {gpu_ids_str} on port {base_port}...")
        try:
            # Execute the command
            subprocess.run(command, shell=shell, check=True)
            logger.info(f"Server started for GPUs {gpu_ids_str} on port {base_port}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start server for GPUs {gpu_ids_str} on port {base_port}: {e}")
            raise


class LLMClientLB:
    """Manages multiple clients to handle API requests, balancing resource usage."""

    def __init__(
        self,
        endpoints: List[Union[int, str]],
        model_name: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        model_path: Optional[str] = None,
        server_port: Optional[int] = None,
        use_docker: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "half",
        enforce_eager: bool = True,
        max_model_len: int = 2048,
        swap_space: int = 4,
        vllm_path: str = "/home/anhvth5/miniconda3/envs/py312-vllm/bin/vllm",
        additional_volumes: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize the Client Load Balancer.

        Args:
            endpoints (List[Union[int, str]]): List of ports or URLs where the API is running.
            model_name (Optional[str]): The model name to use for completions.
            gpu_ids (Optional[List[int]]): List of GPU IDs to use when starting servers.
            model_path (Optional[str]): Path to the model directory.
            base_port (Optional[int]): Base port number to start the server on if starting new.
            use_docker (bool): Whether to use Docker to start the server.
            tensor_parallel_size (int): Size for tensor parallelism.
            gpu_memory_utilization (float): GPU memory utilization fraction.
            dtype (str): Data type to use ("half", "float", etc.).
            enforce_eager (bool): Whether to enforce eager execution.
            max_model_len (int): Maximum model length.
            swap_space (int): Swap space in GB.
            vllm_path (str): Path to the vllm executable.
            additional_volumes (Optional[List[str]]): Additional Docker volume mounts.
            verbose (bool): If True, prints the command instead of executing.
        """
        self.endpoint_usage: Dict[Union[int, str], int] = {endpoint: 0 for endpoint in endpoints}
        self.lock = Lock()
        self.model_name = model_name
        self.clients: Dict[Union[int, str], OpenAI] = {}

        # Validate inputs
        if not endpoints:
            logger.error("No endpoints provided to LLMClientLB.")
            raise ValueError("At least one endpoint must be provided.")

        # Iterate over endpoints and ensure each is available
        for endpoint in endpoints:
            if self._is_endpoint_available(endpoint):
                self.clients[endpoint] = self._initialize_client(endpoint)
                logger.info(f"Connected to existing server at endpoint: {endpoint}")
            else:
                logger.warning(f"Endpoint {endpoint} is not available. Attempting to start server.")

                if gpu_ids is None or model_path is None or server_port is None:
                    logger.error(
                        "To start a new server, gpu_ids, model_path, and base_port must be provided."
                    )
                    raise ValueError(
                        "gpu_ids, model_path, and base_port must be provided to start a new server."
                    )

                try:
                    _start_server(
                        gpu_ids=gpu_ids,
                        model_path=model_path,
                        base_port=server_port,
                        verbose=verbose,
                        use_docker=use_docker,
                        tensor_parallel_size=tensor_parallel_size,
                        gpu_memory_utilization=gpu_memory_utilization,
                        dtype=dtype,
                        enforce_eager=enforce_eager,
                        max_model_len=max_model_len,
                        swap_space=swap_space,
                        vllm_path=vllm_path,
                        additional_volumes=additional_volumes
                    )
                    # Wait for the server to be ready
                    for _ in range(100):
                        if self._is_endpoint_available(endpoint):
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"Server at endpoint {endpoint} did not become available.")

                    self.clients[endpoint] = self._initialize_client(endpoint)
                    logger.info(f"Started and connected to server at endpoint: {endpoint}")
                except Exception as e:
                    logger.error(f"Failed to start server at endpoint {endpoint}: {e}")
                    raise

        # Determine model name if not provided
        if self.model_name is None and endpoints:
            first_client = next(iter(self.clients.values()))
            available_models = first_client.models.list().data
            if available_models:
                self.model_name = available_models[0].id
            else:
                logger.error("No models available from the first client.")
                raise ValueError("No models available from the first client.")

        logger.info(f"Using model: {self.model_name}")

    def _is_endpoint_available(self, endpoint: Union[int, str]) -> bool:
        """
        Checks if the given endpoint is available by sending a health check request.

        Args:
            endpoint (Union[int, str]): The endpoint to check.

        Returns:
            bool: True if available, False otherwise.
        """
        try:
            if isinstance(endpoint, int):
                url = f"http://localhost:{endpoint}/health"
            else:
                url = f"{endpoint}/health"
            response = requests.get(url, timeout=60)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _initialize_client(self, endpoint: Union[int, str]) -> OpenAI:
        """
        Initializes an OpenAI client for the given endpoint.

        Args:
            endpoint (Union[int, str]): The endpoint to connect to.

        Returns:
            OpenAI: An initialized OpenAI client.
        """
        if isinstance(endpoint, int):
            api_base = f"http://localhost:{endpoint}/v1"
        else:
            api_base = endpoint

        api_key = "EMPTY"  # Replace with actual API key if needed
        return OpenAI(api_key=api_key, base_url=api_base)

    def _select_endpoint(self) -> Union[int, str]:
        """
        Select the endpoint with the fewest active requests.

        Returns:
            Union[int, str]: Selected endpoint.
        """
        with self.lock:
            endpoint = min(self.endpoint_usage, key=self.endpoint_usage.get)
            self.endpoint_usage[endpoint] += 1
            return endpoint

    def _release_endpoint(self, endpoint: Union[int, str]):
        """
        Release an endpoint after a request is completed.

        Args:
            endpoint (Union[int, str]): The endpoint to release.
        """
        with self.lock:
            self.endpoint_usage[endpoint] -= 1

    async def create_async(
        self, messages: List[dict], n: int = 1, temperature: float = 0.4, cache: bool = True, max_retries: int = 10, **kwargs
    ) -> Optional[Dict]:
        """
        Create completions using the least busy client asynchronously.

        Args:
            messages (List[dict]): The list of messages for the completion.
            n (int): Number of completions to generate.
            temperature (float): Sampling temperature to use.
            cache (bool): Whether to cache the result.
            max_retries (int): Maximum number of retries if the request fails.
            kwargs: Additional parameters to pass to the completion method.

        Returns:
            Optional[Dict]: Generated completions or None if failed.
        """
        cache_id = identify_uuid([messages, n, temperature, kwargs, self.model_name])
        cache_file = os.path.join(CACHE_DIR, cache_id + ".json")
        if cache and os.path.exists(cache_file):
            try:
                return load_by_ext(cache_file)
            except Exception as e:
                logger.warning(f"Error loading cache file {cache_file}: {e}. Continuing without cache.")

        retries = 0
        while retries < max_retries:
            endpoint = self._select_endpoint()
            client = self.clients.get(endpoint)

            if client is None:
                logger.error(f"No client found for endpoint {endpoint}.")
                retries += 1
                await asyncio.sleep(1)
                continue

            try:
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    **kwargs,
                )
                output = {
                    "input_messages": messages,
                    "choices": [choice.model_dump() for choice in completion.choices]
                }
                if cache:
                    dump_json_or_pickle(output, cache_file)
                return output
            except Exception as e:
                logger.error(f"Error during completion on endpoint {endpoint}: {e}")
                retries += 1
                await asyncio.sleep(5)
            finally:
                self._release_endpoint(endpoint)

        logger.error(f"Failed to create completion after {max_retries} retries.")
        return None

    async def chat(self, input_msg: str, history: List[dict] = [], **kwargs) -> Optional[dict]:
        """
        Facilitates a chat interaction by sending the user's message
        and returning the assistant's response along with updated message history.

        Args:
            input_msg (str): The user's input message.
            history (List[dict]): The prior conversation history.
            kwargs: Additional parameters for the completion.

        Returns:
            Optional[dict]: The assistant's response and updated history, or None if failed.
        """
        msgs = history + [{"role": "user", "content": input_msg}]

        try:
            output = await self.create_async(messages=msgs, **kwargs)
            if output and "choices" in output and len(output["choices"]) > 0:
                assistant_message = output["choices"][0]
                assistant_content = assistant_message["message"]["content"]
                msgs.append({"role": "assistant", "content": assistant_content})
                return {"response": assistant_content, "history": msgs}
            else:
                logger.error("No choices received from model.")
                return None
        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            return None

    async def batch_run_async(
        self,
        batch_messages: List[List[dict]],
        n: int = 1,
        temperature: float = 0.4,
        max_workers: int = 32,
        **kwargs,
    ) -> List[Optional[Dict]]:
        """
        Run multiple completion requests asynchronously with limited concurrency and track progress using tqdm.

        Args:
            batch_messages (List[List[dict]]): List of message lists for each completion request.
            n (int): Number of completions to generate per request.
            temperature (float): Sampling temperature to use.
            max_workers (int): Maximum number of concurrent tasks.
            kwargs: Additional parameters to pass to the completion method.

        Returns:
            List[Optional[Dict]]: A list of completion results.
        """
        semaphore = asyncio.Semaphore(max_workers)

        async def sem_create_async(messages: List[dict]) -> Optional[Dict]:
            async with semaphore:
                return await self.create_async(messages=messages, n=n, temperature=temperature, **kwargs)

        tasks = [sem_create_async(messages) for messages in batch_messages]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing completions")
        return results

    def create(
        self,
        messages: List[dict],
        n: int = 1,
        temperature: float = 0.4,
        cache: bool = True,
        max_retries: int = 10,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Create completions using the least busy client synchronously.

        Args:
            messages (List[dict]): The list of messages for the completion.
            n (int): Number of completions to generate.
            temperature (float): Sampling temperature to use.
            cache (bool): Whether to cache the result.
            max_retries (int): Maximum number of retries if the request fails.
            kwargs: Additional parameters to pass to the completion method.

        Returns:
            Optional[Dict]: Generated completions or None if failed.
        """
        return asyncio.run(self.create_async(messages, n, temperature, cache, max_retries, **kwargs))

    async def create_async_wrapper(self, *args, **kwargs) -> Optional[Dict]:
        return await self.create_async(*args, **kwargs)

    def batch_run(
        self,
        batch_messages: List[List[dict]],
        n: int = 1,
        temperature: float = 0.4,
        max_workers: int = 32,
        **kwargs,
    ) -> List[Optional[Dict]]:
        """
        Run multiple completion requests synchronously using multi-threading.

        Args:
            batch_messages (List[List[dict]]): List of message lists for each completion request.
            n (int): Number of completions to generate per request.
            temperature (float): Sampling temperature to use.
            max_workers (int): Maximum number of concurrent threads.
            kwargs: Additional parameters to pass to the completion method.

        Returns:
            List[Optional[Dict]]: A list of completion results.
        """
        return asyncio.run(self.batch_run_async(batch_messages, n, temperature, max_workers, **kwargs))

    def log_usage(self):
        """Log the current usage of all endpoints."""
        with self.lock:
            logger.info(f"Current endpoint usage: {self.endpoint_usage}")

    @staticmethod
    def get_output_msg(output: Dict) -> str:
        """
        Extract the assistant's message from the output.

        Args:
            output (Dict): The output from the completion.

        Returns:
            str: The assistant's message content.
        """
        return output["choices"][0]["message"]["content"] if output else ""

