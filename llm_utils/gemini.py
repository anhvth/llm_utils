import json
import os
import random
import time

import google.generativeai as genai
from google.generativeai.generative_models import ChatSession
from loguru import logger
from speedy_utils import *

KEYS: list[str] = os.getenv("GEMINI_KEYS").split()


def get_random_key() -> str:
    """Returns a random key from the GEMINI_KEYS environment variable."""
    return random.choice(KEYS)


def _parse_dict(text: str):
    try:
        text = text.split("```json\n")[1].split("```")[0]
        return eval(text)
    except Exception as e:
        raise ValueError(f"Unable to parse Dict: {text}") from e


def parse_response_as_json(raw_response: str):
    """Attempts to parse a response string as JSON."""
    flag = ""
    E = None
    try:
        cleaned = raw_response.replace("```json\n", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        E = e
        flag = "Unable to parse JSON"
    try:
        return _parse_dict(raw_response)
    except Exception as e:
        E = e
        flag = "Unable to parse Dict"
    raise ValueError(f"{flag}: {raw_response}") from E


def get_gemini_response(
    chat_session: ChatSession,
    input_msg: str,
    delay_between_retries: int = 15,
    max_retries: int = 20,
    use_cache: bool = True,
    parse_as_json: bool = True,
    max_parse_failures: int = 1,
):
    """Gets a response from a ChatSession, optionally caching results and parsing JSON."""
    if not isinstance(chat_session, ChatSession):
        raise TypeError(f"Expected ChatSession, got {type(chat_session).__name__}")
    if not isinstance(input_msg, str):
        raise TypeError("input_msg must be a string")

    cache_id = identify(
        [
            str(list(chat_session.history)),
            str(chat_session.model._generation_config),
            input_msg,
            parse_as_json,
        ]
    )
    cache_file = f"~/.cache/gemini_cache/{cache_id}.pkl"
    cache_file = os.path.expanduser(cache_file)

    if os.path.exists(cache_file) and use_cache:
        response = load_json_or_pickle(cache_file)
        return response
    parsed_failures = 0
    for attempt in range(1, max_retries + 1):
        try:
            key = get_random_key()
            genai.configure(api_key=key)
            response = chat_session.send_message(input_msg).text
            if parse_as_json:
                response = parse_response_as_json(response)
            if use_cache:
                dump_json_or_pickle(response, cache_file)
            return response

        except Exception as e:
            if attempt == max_retries:
                raise
            if "429" in str(e) or "500" in str(e):
                logger.warning(
                    f"[gemini] {attempt}/{max_retries} Rate limit key={key[-4:]} exceeded, retrying in {delay_between_retries}s"
                )
                time.sleep(delay_between_retries)
            elif "Unable to parse" in str(e):
                if parsed_failures >= max_parse_failures:
                    logger.error(
                        f"Failed to parse JSON {parsed_failures} times, aborting."
                    )
                    raise
                chat_session.model._generation_config["temperature"] += 0.01
                logger.warning(
                    f"[gemini] JSON parse failed, retrying with higher temperature. {chat_session.model._generation_config["temperature"]}"
                )
                parsed_failures += 1
            else:
                logger.error(f"Error: {str(e)[:100]}")
                raise
