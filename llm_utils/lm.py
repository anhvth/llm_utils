import fcntl
import functools
import json
import logging
import os
import random
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

import dspy
import litellm
import ujson
from dspy import adapters
from dspy.adapters.base import Adapter
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, TrainingJob
from dspy.utils.callback import BaseCallback, with_callbacks
from loguru import logger
from pydantic import BaseModel
from speedy_utils import Clock, dump_json_or_pickle, identify_uuid, load_json_or_pickle


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str | BaseModel


class ChatSession:
    def __init__(
        self,
        lm: "OAI_LM",
        system_prompt: str = None,
        history: List[Message] = [],
        callback=None,
        response_format=None,
    ):
        self.lm = deepcopy(lm)
        self.history = history
        self.callback = callback
        self.response_format = response_format
        if system_prompt:
            system_prompt = {
                "role": "system",
                "content": system_prompt,
            }
            self.history.insert(0, system_prompt)

    def __len__(self):
        return len(self.history)

    def __call__(
        self, text, response_format=None, display=True, max_prev_turns=3, **kwargs
    ) -> str | BaseModel:
        response_format = response_format or self.response_format
        self.history.append({"role": "user", "content": text})
        output = self.lm(
            messages=self.parse_history(), response_format=response_format, **kwargs
        )
        # output could be a string or a pydantic model
        if isinstance(output, BaseModel):
            self.history.append({"role": "assistant", "content": output})
        else:
            assert response_format is None
            self.history.append({"role": "assistant", "content": output})
        if display:
            self.inspect_history(max_prev_turns=max_prev_turns)

        if self.callback:
            self.callback(self, output)
        return output

    def parse_history(self, indent=None):
        parsed_history = []
        for m in self.history:
            if isinstance(m["content"], str):
                parsed_history.append(m)
            elif isinstance(m["content"], BaseModel):
                parsed_history.append(
                    {
                        "role": m["role"],
                        "content": m["content"].model_dump_json(indent=indent),
                    }
                )
            else:
                raise ValueError(f"Unexpected content type: {type(m['content'])}")
        return parsed_history

    def inspect_history(self, max_prev_turns=3):
        from llm_utils import display_chat_messages_as_html

        h = self.parse_history(indent=2)
        try:
            from IPython.display import clear_output

            clear_output()
            display_chat_messages_as_html(h[-max_prev_turns * 2 :])
        except:
            pass


class OAI_LM(dspy.LM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str = None,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[Any] = None,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if model is None:
            model = self.list_models(kwargs.get("base_url"))[0]
            model = f"openai/{model}"
            # logger.info(f"Using default model: {model}")

        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,  # disable cache handling by default and implement custom cache handling
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            launch_kwargs=launch_kwargs,
            **kwargs,
        )
        self.do_cache = cache

    def __call__(
        self,
        prompt=None,
        messages=None,
        response_format: BaseModel = None,
        cache=None,
        retry_count=0,
        **kwargs,
    ) -> str | BaseModel:
        if retry_count > self.kwargs.get("num_retries", 3):
            raise ValueError("Retry limit exceeded")
        id = None
        cache = cache if cache is not None else self.do_cache
        if response_format:
            assert issubclass(
                response_format, BaseModel
            ), f"response_format must be a pydantic model, {type(response_format)} provided"
        result = None
        if cache:
            _kwargs = {**self.kwargs, **kwargs}

            s = str(
                [
                    prompt,
                    messages,
                    (response_format.model_json_schema() if response_format else None),
                    _kwargs["temperature"],
                    _kwargs["max_tokens"],
                    self.model,
                ]
            )
            id = identify_uuid(s)
            result = self.load_cache(id)
        if not result:
            try:
                result = super().__call__(
                    prompt=prompt,
                    messages=messages,
                    **kwargs,
                    response_format=response_format,
                )
                if kwargs.get("n", 1) == 1:
                    result = result[0]
            except litellm.exceptions.ContextWindowExceededError as e:
                logger.error(f"Context window exceeded: {e}")
            except litellm.exceptions.APIError as e:
                # sleep for a random time between 1 and 5 seconds
                logger.error(f"API Error: {e}")
                time.sleep(random.randint(1, 3))
                return self.__call__(
                    prompt=prompt,
                    messages=messages,
                    response_format=response_format,
                    cache=cache,
                    retry_count=retry_count + 1,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                raise

        if self.do_cache:
            self.dump_cache(id, result)
        if response_format:
            import json_repair

            try:
                return response_format(**json_repair.loads(result))
            except Exception as e:
                raise ValueError(f"Failed to parse response for {response_format}: {e}")
        return result

    def get_session(
        self,
        system_prompt,
        history: List[Message] = None,
        callback=None,
        response_format=None,
        **kwargs,
    ) -> ChatSession:
        if history is None:
            history = []
        else:
            history = deepcopy(history)
        return ChatSession(
            self,
            system_prompt=system_prompt,
            history=history,
            callback=callback,
            response_format=response_format,
            **kwargs,
        )

    def dump_cache(self, id, result):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)

            dump_json_or_pickle(result, cache_file)
        except Exception as e:
            logger.warning(f"Cache dump failed: {e}")

    def load_cache(self, id):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)
            if not os.path.exists(cache_file):
                return
            return load_json_or_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def list_models(self, base_url):
        import openai

        base_url = base_url or self.kwargs["base_url"]
        client = openai.OpenAI(base_url=base_url)
        page = client.models.list()
        return [d.id for d in page.data]

    @property
    def client(self):
        import openai

        return openai.OpenAI(base_url=self.kwargs["base_url"])


# If these aren't accessible from your context, bring them in as needed
# from your_package import OAI_LM, Clock

# logger = logging.getLogger(__name__)


class FileLock:
    """
    A simple context manager for POSIX file locking.
    """

    def __init__(self, lock_path):
        self.lock_path = lock_path
        self.fd = None

    def __enter__(self):
        # Open the lock file in write mode, creating it if necessary
        self.fd = open(self.lock_path, "w")
        # Acquire an exclusive lock on the file
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release the lock
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()
        self.fd = None


class OAI_LMs(OAI_LM):
    def __init__(self, ports, *args, **kwargs):
        self.ports = ports

        # Figure out the default port from the passed base_url
        self.default_port = kwargs["base_url"].split(":")[-1].replace("/v1", "")

        # Build a list of base URLs for each port
        self.base_urls = [
            kwargs["base_url"].replace(self.default_port, str(port)) for port in ports
        ]

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Path to the lock file and the JSON that tracks in-flight counts
        # Make sure these paths are writable.
        self._lock_file_path = "/tmp/oai_lms_filelock.lock"
        self._counts_file_path = "/tmp/oai_lms_in_flight_counts.json"

        # Ensure that the in-flight counts file exists with default zeros
        if not os.path.exists(self._counts_file_path):
            self._write_in_flight_counts({url: 0 for url in self.base_urls})

        self.clock = Clock()

    def __call__(self, *args, **kwargs):
        """
        Select the URL with the fewest active requests, increment its count,
        call the API, then decrement the count.
        """
        # --- ACQUIRE LOCK, PICK TARGET URL, INCREMENT COUNT ---
        with FileLock(self._lock_file_path):
            in_flight = self._read_in_flight_counts()
            # Choose the URL with the fewest active (in-flight) requests
            target_url = min(in_flight, key=in_flight.get)
            in_flight[target_url] += 1
            self._write_in_flight_counts(in_flight)

        # Update the base_url for the request
        kwargs["base_url"] = target_url

        try:
            # Perform the actual call
            result = super().__call__(*args, **kwargs)
        finally:
            # --- ACQUIRE LOCK, DECREMENT COUNT ---
            with FileLock(self._lock_file_path):
                in_flight = self._read_in_flight_counts()
                in_flight[target_url] -= 1
                self._write_in_flight_counts(in_flight)

                # Optionally log the in-flight counts every ~10 seconds
                if self.clock.time_since_last_checkpoint() > 10:
                    logger.debug(f"In-flight request counts: {in_flight}")
                    self.clock.tick()

        return result

    def _read_in_flight_counts(self):
        """Read the JSON file that tracks in-flight counts."""
        if not os.path.exists(self._counts_file_path):
            return {url: 0 for url in self.base_urls}
        with open(self._counts_file_path, "r") as f:
            return json.load(f)

    def _write_in_flight_counts(self, data):
        """Write the in-flight counts dictionary to JSON."""
        with open(self._counts_file_path, "w") as f:
            json.dump(data, f)
