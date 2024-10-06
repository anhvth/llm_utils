import asyncio
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Union
from speedy_utils import identify_uuid, dump_json_or_pickle, load_by_ext
from loguru import logger
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio

# ~/.cache/llm_cache/
CACHE_DIR = str(Path('~/.cache/llm_cache').expanduser().resolve())

class LLMClientLB:
    """Manages multiple clients to handle API requests, balancing resource usage."""

    def __init__(self, endpoints: List[Union[int, str]], model_name: str = None):
        """
        Initialize the ClientManager.

        Args:
            endpoints (List[Union[int, str]]): List of ports or URLs where the API is running.
            model_name (str): The model name to use for completions.
        """
        self.endpoint_usage = {endpoint: 0 for endpoint in endpoints}
        self.lock = Lock()
        self.model_name = model_name
        self.clients = {endpoint: self._initialize_client(endpoint) for endpoint in endpoints}
        if self.model_name is None and len(endpoints) > 0:
            self.model_name = self.clients[endpoints[0]].models.list().data[0].id

        logger.info(f"Using model {self.model_name}")

    def _initialize_client(self, endpoint: Union[int, str]):
        """Initialize a client for a given endpoint."""
        if isinstance(endpoint, int) or (isinstance(endpoint, str) and endpoint.isdigit()):
            # Treat as a port number
            api_base = f"http://localhost:{endpoint}/v1"
        else:
            api_base = endpoint

        api_key = "EMPTY"  # Replace with actual API key if needed
        return OpenAI(api_key=api_key, base_url=api_base)

    def _select_endpoint(self) -> Union[int, str]:
        """Select the endpoint with the fewest active requests."""
        with self.lock:
            endpoint = min(self.endpoint_usage, key=self.endpoint_usage.get)
            self.endpoint_usage[endpoint] += 1
            return endpoint

    def _release_endpoint(self, endpoint: Union[int, str]):
        """Release an endpoint after a request is completed."""
        with self.lock:
            self.endpoint_usage[endpoint] -= 1

    async def create_async(self, messages: List[dict], n: int = 1, temperature=0.4, cache=True, max_retries=10, **kwargs) -> List[str]:
        """
        Create completions using the least busy client asynchronously.

        Args:
            messages (List[dict]): The list of messages for the completion.
            n (int): Number of completions to generate.
            temperature (float): Sampling temperature to use.
            cache (bool): Whether to cache the result.
            max_retries (int): Maximum number of retries if the request fails.

        Returns:
            List[str]: Generated completions.
        """
        cache_id = identify_uuid([messages, n, temperature, kwargs])
        cache_file = os.path.join(CACHE_DIR, cache_id + ".json")
        if cache and os.path.exists(cache_file):
            return load_by_ext(cache_file)

        retries = 0
        while retries < max_retries:
            endpoint = self._select_endpoint()
            client = self.clients[endpoint]

            try:
                completion = await asyncio.to_thread(client.chat.completions.create,  # Run sync method in thread
                                                     model=self.model_name,
                                                     messages=messages,
                                                     temperature=temperature,
                                                     n=n,
                                                     **kwargs)
                output = [choice.message.content for choice in completion.choices]
                if cache:
                    logger.debug(f"Caching completion: {cache_id}")
                    dump_json_or_pickle(output, cache_file)
                return output
            except Exception as e:
                logger.error(f"Error during completion on endpoint {endpoint}: {e}")
                retries += 1
                await asyncio.sleep(5)  # Wait for 5 seconds before retrying
            finally:
                self._release_endpoint(endpoint)

        raise Exception(f"Failed to create completion after {max_retries} retries.")

    def create(self, messages: List[dict], n: int = 1, temperature=0.4, **kwargs) -> List[str]:
        """
        Create completions using the least busy client.

        Args:
            messages (List[dict]): The list of messages for the completion.
            n (int): Number of completions to generate.

        Returns:
            List[str]: Generated completions.
        """
        return asyncio.run(self.create_async(messages, n, temperature, **kwargs))

    def log_usage(self):
        """Log the current usage of all endpoints."""
        with self.lock:
            logger.info(f"Current endpoint usage: {self.endpoint_usage}")

    async def batch_run(self, batch_messages: List[List[dict]], n: int = 1, temperature=0.4, **kwargs) -> List[List[str]]:
        """
        Run multiple completion requests asynchronously and track progress using tqdm.

        Args:
            batch_messages (List[List[dict]]): List of lists of messages, where each sublist corresponds to a single completion request.
            n (int): Number of completions to generate per request.
            temperature (float): Sampling temperature to use.
            kwargs: Additional parameters to pass to the completion method.

        Returns:
            List[List[str]]: A list of lists of generated completions.
        """
        tasks = [
            self.create_async(messages=messages, n=n, temperature=temperature, **kwargs)
            for messages in batch_messages
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing completions")
        return results