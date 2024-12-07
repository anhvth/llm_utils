from collections import Counter, defaultdict
from typing import Dict, List, overload
import re
import openai
from loguru import logger
from openai import OpenAI
from speedy_utils import Clock, multi_thread
from transformers import AutoTokenizer
from speedy_utils.all import *
from openai.types.chat import ChatCompletion
from openai.types import Completion

# Ensure you have the OpenAI library installed:
# pip install openai

# Known models mapping
KNOWN_MODELS = {}
clock = Clock()


class OpenAIWraper:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )

    def __init__(
        self,
        model,
        base_port: int = 2800,
        ping_range: int = 33,
        max_tokens: int = 2048,
        cache: bool = True,
        **kwargs,
    ):
        """
        Initializes the LMGenerationMultiWorker with the specified model and scans for available OpenAI workers.

        Args:
            model (str): The model name to use.
            base_port (int): The starting port number to scan for workers.
            ping_range (int): The range of ports to scan from the base_port.
            max_tokens (int): The maximum number of tokens per request.
            **kwargs: Additional keyword arguments for the OpenAI client.
        """
        self.kwargs = kwargs
        # self._history = []
        model_name = KNOWN_MODELS.get(model, model)
        self.scanned_ports = self.scan_model_ports(
            base_port=base_port, ping_range=ping_range, verbose=False
        )
        ports = self.scanned_ports.get(model_name, [-1])
        self.counter = Counter({port: 0 for port in ports})
        self.max_tokens = max_tokens
        self.temperature = 0
        self.model = f"{model_name}"
        self.cache = cache

        self.clients = {
            port: OpenAI(base_url=f"http://localhost:{port}/v1", **kwargs)
            for port in ports
        }

    def get_client(self):
        """
        Selects the least used OpenAI client based on the current counter.

        Returns:
            A tuple of the selected OpenAI client and its corresponding port.
        """
        port, _ = self.counter.most_common()[-1]
        client = self.clients.get(port)
        if client is None:
            logger.error(f"No client found for port {port}.")
            raise ValueError(f"No client found for port {port}.")
        return client, port

    def count_tokens(self, messages) -> int:
        """
        Counts the number of tokens in the given messages.

        Args:
            messages (list or str): The input messages or prompt.

        Returns:
            The total number of tokens.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list) or not isinstance(messages[0], dict):
            messages = [{"role": "user", "content": str(messages)}]
        return len(self.tokenizer.apply_chat_template(messages))

    def chat(
        self,
        messages=None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Completes a chat prompt with the selected OpenAI worker.

        Args:
            messages (list of dict, optional): Chat messages for chat-based models.
            **kwargs: Additional parameters for the OpenAI chat completion.

        Returns:
            The generated chat completion.
        """
        if not messages:
            raise ValueError("Messages must be provided for chat completion.")

        if not "max_tokens" in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        if not "temperature" in kwargs:
            kwargs["temperature"] = self.temperature

        try:
            client, port = self.get_client()
            self.counter[port] += 1
            input_data = {
                "model": self.model,
                "messages": messages,
                **kwargs,
            }

            def _run_with_cache(input_data):
                response = client.chat.completions.create(**input_data)
                return response

            if self.cache:
                _run_with_cache = memoize(_run_with_cache)

            _run_with_cache = imemoize(_run_with_cache)

            response = _run_with_cache(input_data)
            # self._history.append(
            #     {
            #         "messages": messages,
            #         "response": response,
            #     }
            # )
            return response

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error during chat completion: {e}")
            raise
        finally:
            self.counter[port] -= 1
            if clock.time_since_last_checkpoint() > 5:
                logger.info(f"Request on port={port} completed | {self.counter} ")
                clock.tick()

    def generate(
        self,
        prompt=None,
        **kwargs,
    ) -> Completion:
        """
        Sends a request to the selected OpenAI worker and returns the response.

        Args:
            prompt (str, optional): The prompt to generate completions for.
            **kwargs: Additional parameters for the OpenAI completion.

        Returns:
            The generated completion.
        """
        if not prompt:
            raise ValueError("Prompt must be provided for text completion.")

        if not "max_tokens" in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        if not "temperature" in kwargs:
            kwargs["temperature"] = self.temperature

        try:
            client, port = self.get_client()
            self.counter[port] += 1
            input_data = {
                "model": self.model,
                "prompt": prompt,
                **kwargs,
            }

            def _run_with_cache(input_data):
                response = client.completions.create(**input_data)
                return response

            if self.cache:
                _run_with_cache = memoize(_run_with_cache)

            response = _run_with_cache(input_data)
            # self._history.append(
            #     {
            #         "prompt": prompt,
            #         "response": response,
            #     }
            # )
            return response

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error during text completion: {e}")
            raise
        finally:
            self.counter[port] -= 1
            if clock.time_since_last_checkpoint() > 5:
                logger.info(f"Request on port={port} completed | {self.counter}")
                clock.tick()

    def __call__(
        self,
        prompt=None,
        messages=None,
        **kwargs,
    ) -> Completion:
        """
        Sends a request to the selected OpenAI worker and returns the response.

        Args:
            prompt (str, optional): The prompt to generate completions for.
            messages (list of dict, optional): Chat messages for chat-based models.
            **kwargs: Additional parameters for the OpenAI completion.

        Returns:
            The generated completion.
        """
        if messages:
            return self.chat(messages=messages, **kwargs)
        return self.generate(prompt=prompt, **kwargs)

    @staticmethod
    def scan_model_ports(
        base_port: int = 2800, ping_range: int = 20, verbose: bool = True
    ) -> Dict[str, List[int]]:
        """
        Scans the local servers to find available OpenAI workers and maps models to their ports.

        Args:
            base_port (int): The starting port number to scan.
            ping_range (int): The range of ports to scan from the base_port.

        Returns:
            A dictionary mapping model names to lists of ports where they are available.
        """
        model_to_servers: Dict[str, List[int]] = defaultdict(list)
        server_ports = [base_port + i for i in range(ping_range)]

        def scan_port(port):
            client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", timeout=1)
            try:
                models = client.models.list()
                return {model.id: port for model in models}
            except Exception as e:
                # logger.error(f"Error scanning port {port}: {e}")
                return {}

        results = multi_thread(
            scan_port, server_ports, workers=ping_range, verbose=False
        )

        for result in results:
            for model, port in result.items():
                model_to_servers[model].append(port)
        if verbose:
            for model, ports in model_to_servers.items():
                print(f"Model: {model} | Ports: {ports}")
        return model_to_servers

    def ih(self, i=-1):
        """
        Inspects the history at index i and returns an HTML representation of the chat messages.

        Args:
            i (int): The index in the history to inspect.

        Returns:
            HTML representation of the chat messages.
        """
        from llm_utils import display_chat_messages_as_html, inspect_msgs

        res = self._history[i]["response"].choices[0].message.content
        msgs = self._history[i]["messages"] + [{"role": "assistant", "content": res}]
        return display_chat_messages_as_html(msgs)

    def response(self, i):
        """
        Retrieves the response content from the history at index i.

        Args:
            i (int): The index in the history.

        Returns:
            The response content.
        """
        res = self._history[i]["response"].choices[0].message.content
        return res

    def count_history_tokens(self, i):
        """
        Counts the tokens in the history at index i.

        Args:
            i (int): The index in the history.

        Returns:
            The total number of tokens.
        """
        res = self._history[i]["response"].choices[0].message.content
        msgs = self._history[i]["messages"] + [{"role": "assistant", "content": res}]
        return self.count_tokens(msgs)

    @staticmethod
    def apply_template(messages, assistant_prefix="", auto_system=False):
        """
        Applies a chat template to the messages.

        Args:
            messages (list of dict): The chat messages.
            assistant_prefix (str): An optional prefix for the assistant's response.
            auto_system (bool): Whether to automatically include the system message.

        Returns:
            The templated chat messages with the assistant prefix.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        prompt = (
            OpenAIWraper.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + assistant_prefix
        )

        if auto_system:
            return prompt

        if messages[0].get("role") != "system":
            spliter = "<|im_end|>\n"
            idx = prompt.index(spliter) + len(spliter)
            prompt = prompt[idx:]
        return prompt

    @staticmethod
    def chatmltext_to_messages(chatmltext):
        """
        Converts a chatML text to a list of chat messages.

        Args:
            chatmltext (str): The chatML text.

        Returns:
            A list of chat messages.
        """
        # format <|im_start>role\ncontent<|im_end>\n<|im_start>role\ncontent<|im_end>

        pattern = re.compile(r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>", re.DOTALL)
        matches = pattern.findall(chatmltext)
        messages = [
            {"role": role.strip(), "content": content.strip()}
            for role, content in matches
        ]
        return messages

    # @property
    # def history(self, i=-1):
    #     item = self._history[i]
    #     if "prompt" in item:
    #         prompt = item["prompt"]
    #         response = item["response"].choices[0].message.content
    #         # if more choices warning
    #         if len(item["response"].choices) > 1:
    #             logger.warning(f"More than one choice found in history at index {i}.")
    #         conv = prompt + response.strip()
    #         if not conv.endswith("<|im_end|>"):
    #             conv += "<|im_end|>"

    #         return self.chatmltext_to_messages(conv)
    #     elif "messages" in item:
    #         response = item["response"].choices[0].message.content
    #         if len(item["response"].choices) > 1:
    #             logger.warning(f"More than one choice found in history at index {i}.")
    #         return item["messages"] + [{"role": "assistant", "content": response}]
    def complete_with_prefix(
        self,
        prompt: str = None,
        msgs: List[str] = None,
        prefix='',
        verbose=False,
        retrun_msgs=False,
        **kwargs,
    ) -> List[str]:
        msgs = [{"role": "user", "content": prompt}] if prompt else msgs
        prompt_lm = self.apply_template(msgs, assistant_prefix=prefix)
        if verbose:
            print(f"Prompt LM:\n```{prompt_lm}```")
        choices = self.generate(prompt_lm, **kwargs).choices
        rets = []
        for choice in choices:
            ai_msg = prefix + choice.text
            if choice.stop_reason == 151643:
                ai_msg += "<|endoftext|>"
            elif choice.stop_reason:
                ai_msg += choice.stop_reason
            rets.append(ai_msg)
        if retrun_msgs:
            return rets, self.chatmltext_to_messages(prompt_lm)
        return rets
