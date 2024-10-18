import os
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generic, List, Literal, Optional, TypeVar

import openai
import psutil
from loguru import logger
from openai.types.completion import Completion
from pydantic import BaseModel
from speedy_utils import dump_json_or_pickle, identify_uuid, load_by_ext

T = TypeVar('T')

class LLMClientLB(Generic[T]):
    def __init__(self, workers: List[T], default_model_name, cache_dir: Optional[Path] = None,
                 debug: bool = True):
        """
        Initialize with a list of workers and task distribution logic.
        :param workers: List of worker instances (OpenAI clients).
        :param default_model_name: Default model name for OpenAI client.
        :param cache_dir: Optional cache directory.
        :param debug: If True, enables debug-level logging.
        """
        self.workers = workers
        self.usage_counter = Counter()
        for worker in workers:
            self.usage_counter[worker] = 0
        self.lock = Lock()
        self.default_model_name = default_model_name
        self.histories = []
        self.cache_dir = cache_dir or Path(os.path.expanduser("~/.cache/llm_cache"))
        self.debug = debug

        if self.debug:
            logger.enable(__name__)
            logger.debug("Logger initialized in debug mode.")
        else:
            logger.disable(__name__)

    def _get_least_busy_worker(self) -> T:
        """Select the worker that has handled the fewest tasks."""
        with self.lock:
            v2k = {v: k for k, v in self.usage_counter.items()}
            min_value = min(v2k.keys())
            worker = v2k[min_value]
            logger.info(f"Selected worker: {worker}, usage_counter: {self.usage_counter}")
            return worker

    def _update_worker_usage(self, worker: T, delta: int):
        """Update the usage count of a worker."""
        with self.lock:
            self.usage_counter[worker] += delta
            action = "Incremented" if delta > 0 else "Decremented"
            logger.debug(f"{action} usage count for worker {worker}, Usage: {self.usage_counter}")

    def _process_response(self, completion: dict, msgs, response_types):
        """Helper method to process the response and extract content based on response_types."""
        result = {}
        content = completion["choices"][0]["message"]["content"]
        if "content" in response_types:
            result["content"] = content

        history = msgs.copy() + [{"role": "assistant", "content": content}]
        if self.debug:
            self.histories.append(history)

        if "history" in response_types:
            result["history"] = history

        if "completion" in response_types:
            result["completion"] = completion

        if "response_msg" in response_types:
            result["response_msg"] = {"role": "assistant", "content": content}

        return result

    def converse(
        self,
        prompt: str,
        history=None,
        temperature=0.01,
        n=1,
        response_types: List[Literal["completion", "content", "history", "response_msg"]] = [
            "response_msg",
            "history",
            "content",
        ],
        cache=True,
        system_msg: str = None,
        **kwargs
    ):
        """Handles synchronous conversation with task distribution."""
        assert not (history and system_msg), "Provide either history or system_msg, not both."

        assert n == 1, "n > 1 is not supported"
        if system_msg:
            history = [{"role": "system", "content": system_msg}]
        if not history:
            history = []

        msgs = history + [{"role": "user", "content": prompt}]
        id = identify_uuid([prompt, history, kwargs])
        cache_file = self.cache_dir / f"{id}.json"

        # Check if the response is cached
        completion = None
        if cache and os.path.exists(cache_file):
            try:
                logger.info(f"Loading cached completion from {cache_file}")
                completion = load_by_ext(cache_file)
            except:
                logger.warning(f"Failed to load cached completion from {cache_file}, now generating a new one.")

        if not completion:
            client = self._get_least_busy_worker()
            self._update_worker_usage(client, 1)

            completion = client.chat.completions.create(
                model=self.default_model_name, messages=msgs, temperature=temperature, **kwargs
            ).model_dump()

            self._update_worker_usage(client, -1)

            if cache:
                logger.info(f"Caching completion to {cache_file}")
                dump_json_or_pickle(completion, cache_file)

        # Process the response
        return self._process_response(completion, msgs, response_types)

    async def aconverse(
        self,
        prompt: str,
        history=None,
        temperature=0.01,
        n=1,
        response_types: List[Literal["completion", "content", "history", "response_msg"]] = [
            "response_msg",
            "history",
            "content",
        ],
        cache=True,
        system_msg: str = None,
        **kwargs,
    ):
        """Handles asynchronous conversation with task distribution."""
        assert not (history and system_msg), "Provide either history or system_msg, not both."

        assert n == 1, "n > 1 is not supported"
        if system_msg:
            history = [{"role": "system", "content": system_msg}]
        if not history:
            history = []

        msgs = history + [{"role": "user", "content": prompt}]
        id = identify_uuid([prompt, history, kwargs])
        cache_file = self.cache_dir / f"{id}.json"

        # Check if the response is cached
        completion = None
        if cache and os.path.exists(cache_file):
            try:
                logger.info(f"Loading cached completion from {cache_file}")
                completion = load_by_ext(cache_file)
            except:
                logger.warning(f"Failed to load cached completion from {cache_file}, now generating a new one.")

        if not completion:
            client = self._get_least_busy_worker()
            self._update_worker_usage(client, 1)

            completion = await client.chat.completions.create(
                model=self.default_model_name, messages=msgs, temperature=temperature, **kwargs
            )
            completion = completion.model_dump()

            self._update_worker_usage(client, -1)

            if cache:
                logger.info(f"Caching completion to {cache_file}")
                dump_json_or_pickle(completion, cache_file)

        # Process the response
        return self._process_response(completion, msgs, response_types)

    @classmethod
    def auto_create(cls, ports=[], **kwargs):
        """Automatically create LLMClientLB based on detected vLLM ports."""
        if not ports:
            ports = get_vllm_ports() + ports
        base_urls = [f"http://localhost:{port}/v1" for port in ports]
        model_name, clients = None, []

        def get_model(client):
            return client.models.list().data[0].id

        for base_url in base_urls:
            client = openai.OpenAI(base_url=base_url)
            current_model_name = get_model(client)
            model_name = model_name or current_model_name
            assert current_model_name == model_name, f"New model: {current_model_name} != {model_name}"
            clients.append(client)

        return cls(clients, model_name, **kwargs)

    @classmethod
    def openai_create(cls, model_name="gpt-4o-mini"):
        """Create an instance with OpenAI's public API."""
        client = openai.OpenAI()
        return cls([client], model_name)

    def inspect_history(self, k=1, num_last_per_history=100):
        """Inspect history of past conversations (debug mode only)."""
        assert self.debug, "Must be in debug mode to inspect history"
        for i, history in enumerate(self.histories[-k:]):
            # print(f"Conversation {i}: ==========")
            from llm_utils import display_chat_messages_as_html
            display_chat_messages_as_html(history[-num_last_per_history:])

    async def aconverse(self, **kwargs):
        """Asynchronous version of converse."""
        return self.converse(**kwargs)
    
    async def batch_aconverse(self, prompts, **kwargs):
        """Asynchronous version of batch_converse."""
    