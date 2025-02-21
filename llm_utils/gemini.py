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


def parse_response_as_json(raw_response: str):
    """Attempts to parse a response string as JSON."""
    try:
        cleaned = raw_response.replace("```json\n", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Unable to parse JSON: {raw_response}") from e


def get_response(
    chat_session: ChatSession,
    input_msg: str,
    delay_between_retries: int = 15,
    max_retries: int = 20,
    use_cache: bool = True,
    parse_as_json: bool = True,
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
            use_cache,
        ]
    )
    cache_file = f"~/.cache/gemini_cache/{cache_id}.pkl"

    if os.path.exists(cache_file) and use_cache:
        response = load_json_or_pickle(cache_file)
        return parse_response_as_json(response) if parse_as_json else response

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
            if "429" in str(e):
                logger.warning(
                    f"[gemini] Rate limit key={key[-4:]} exceeded, retrying in {delay_between_retries}s"
                )
                time.sleep(delay_between_retries)
            elif "Unable to parse JSON" in str(e):
                logger.warning(
                    "[gemini] JSON parse failed, retrying with higher temperature."
                )
                chat_session.model._generation_config["temperature"] = 1
            else:
                logger.error(f"Error: {e}")
                raise
