import functools
import logging
import os
import threading
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
from speedy_utils import dump_json_or_pickle, identify_uuid, load_json_or_pickle


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
            logger.info(f"Using default model: {model}")

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
        **kwargs,
    ) -> str | BaseModel:
        id = None
        cache = cache if cache is not None else self.do_cache
        if response_format:
            assert issubclass(
                response_format, BaseModel
            ), f"response_format must be a pydantic model, {type(response_format)} provided"
        result = None
        if cache:
            # max_tokens = kwargs.get("max_tokens", self.max_tokens)
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
                if kwargs.get('n', 1) == 1:
                    result = result[0]
            except litellm.exceptions.ContextWindowExceededError as e:
                logger.error(f"Context window exceeded: {e}")
                # raise e
            except Exception as e:
                logger.error(f"Error: {e}")
                # raise e

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