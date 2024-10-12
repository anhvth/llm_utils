import asyncio
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union
import requests
from loguru import logger
from openai import OpenAI
from speedy_utils import dump_json_or_pickle, identify_uuid, load_by_ext
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import psutil

# Define the cache directory
CACHE_DIR = str(Path("~/.cache/llm_cache").expanduser().resolve())
os.makedirs(CACHE_DIR, exist_ok=True)

def get_vllm_ports():
    """Find all ports being used by VLLM processes."""
    vllm_ports = []

    # Iterate through all processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process name or command line has 'vllm'
            if 'vllm' in ' '.join(proc.info['cmdline']).lower():
                # Extract the port number from the command line arguments
                for arg in proc.info['cmdline']:
                    if '--port' in arg:
                        port_index = proc.info['cmdline'].index(arg) + 1
                        if port_index < len(proc.info['cmdline']):
                            port = proc.info['cmdline'][port_index]
                            vllm_ports.append(port)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return [int(x) for x in vllm_ports]


class LLMClientLoadBalancer:
    """Manages multiple clients to handle API requests with load balancing."""

    def __init__(self, endpoints: List[int] = None):
        if endpoints is None:
            endpoints = get_vllm_ports()
        self.endpoint_usage = {endpoint: 0 for endpoint in endpoints}
        self.lock = Lock()
        self.clients = self._initialize_clients(endpoints)
        self.model_name = self._get_model_name_from_clients()

    def _initialize_clients(self, endpoints: List[int]) -> Dict[int, OpenAI]:
        """Initialize available clients based on the endpoints (ports)."""
        clients = {}
        for endpoint in tqdm(endpoints, desc="Initializing Clients", unit="client"):
            if self._is_endpoint_available(endpoint):
                clients[endpoint] = self._create_openai_client(endpoint)
            else:
                logger.warning(f"Endpoint {endpoint} is not available and will be skipped.")
        return clients

    def _endpoint_to_url(self, endpoint: int) -> str:
        """Convert a port endpoint to its corresponding URL."""
        return f"http://localhost:{endpoint}/"

    def _is_endpoint_available(self, endpoint: int) -> bool:
        """Check if the given endpoint is available."""
        url = self._endpoint_to_url(endpoint)
        try:
            response = requests.get(url + "health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _create_openai_client(self, endpoint: int) -> OpenAI:
        """Initialize the OpenAI client for the given endpoint."""
        api_base = self._endpoint_to_url(endpoint) + "v1"
        return OpenAI(api_key="EMPTY", base_url=api_base)

    def _get_model_name_from_clients(self) -> str:
        """Get the model name from the first available client."""
        first_client = next(iter(self.clients.values()), None)
        if first_client:
            models = first_client.models.list().data
            if models:
                return models[0].id
        logger.error("No models found from the client endpoints.")
        raise ValueError("No available models.")

    def _select_least_busy_endpoint(self) -> int:
        """Select the endpoint with the fewest active requests."""
        with self.lock:
            return min(self.endpoint_usage, key=self.endpoint_usage.get)

    def _update_usage_count(self, endpoint: int, delta: int):
        """Update the usage count of a specific endpoint."""
        with self.lock:
            self.endpoint_usage[endpoint] += delta


class LLMClientLoadBalancerAsync(LLMClientLoadBalancer):
    """Extends LLMClientLoadBalancer with asynchronous capabilities."""

    def __init__(self, endpoints: List[int] = None):
        super().__init__(endpoints)
        self.cache_hits = 0
        self.cache_misses = 0

    async def create_async(
        self,
        messages: List[dict] | str,
        n: int = 1,
        temperature: float = 0.4,
        cache: bool = True,
        max_retries: int = 10,
        **kwargs,
    ) -> Optional[Dict]:
        """Create completions using the least busy client asynchronously."""

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        cache_id = identify_uuid([messages, n, temperature, kwargs, self.model_name])
        cache_file = f"{CACHE_DIR}/{cache_id}.json"
        if cache and os.path.exists(cache_file):
            try:
                logger.debug(f"Cache hit: {cache_file}")
                self.cache_hits += 1  # Increment cache hit counter
                return load_by_ext(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, computing normally.")

        self.cache_misses += 1  # Increment cache miss counter
        err = None
        for attempt in range(max_retries):
            endpoint = self._select_least_busy_endpoint()
            client = self.clients.get(endpoint)
            if not client:
                logger.error(f"No client for endpoint {endpoint}. Retrying {attempt + 1}/{max_retries}.")
                await asyncio.sleep(1)
                continue
            try:
                self._update_usage_count(endpoint, 1)
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    **kwargs,
                )
                result = {"input_messages": messages, "choices": [choice.model_dump() for choice in completion.choices]}
                if cache:
                    dump_json_or_pickle(result, cache_file)
                    logger.debug(f"Cache saved: {cache_file}")
                return result
            except Exception as e:
                logger.error(f"Error during async completion on endpoint {endpoint}: {e}")
                await asyncio.sleep(1)
                err = e
            finally:
                self._update_usage_count(endpoint, -1)
        logger.error("Max retries reached without success.")
        return err

    async def batch_run_async(
        self, batch_messages: List[List[dict]], n: int = 1, temperature: float = 0.4, max_workers: int = 32, **kwargs
    ) -> List[Optional[Dict]]:
        """Run multiple completion requests asynchronously with limited concurrency, preserving order."""
        semaphore = asyncio.Semaphore(max_workers)

        async def wrapped_create(index: int, messages: List[dict]):
            async with semaphore:
                result = await self.create_async(messages=messages, n=n, temperature=temperature, **kwargs)
                return index, result

        with tqdm_asyncio(total=len(batch_messages), desc="Processing Completions", unit="req") as pbar:
            # Create tasks, associating each one with its index
            tasks = [wrapped_create(i, messages) for i, messages in enumerate(batch_messages)]
            results = [None] * len(batch_messages)  # Placeholder for results

            # Process results as they complete, updating the results list
            for coro in asyncio.as_completed(tasks):
                index, result = await coro
                results[index] = result
                pbar.update(1)
                pbar.set_postfix({"hits": self.cache_hits, "misses": self.cache_misses})

            return results
