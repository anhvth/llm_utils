import json
import os
import random
import time

import google.generativeai as genai
from google.generativeai import ChatSession
from loguru import logger
from speedy_utils import *
import ast


def get_random_key() -> str:
    """Returns a random key from the GEMINI_KEYS environment variable."""
    try:
        KEYS: list[str] = os.getenv("GEMINI_KEYS").split('|')
    except AttributeError:
        KEYS = [""]
        logger.warning("No GEMINI_KEYS environment variable found. Gemini API will not work.")
    return random.choice(KEYS)


def _extract_json_code_block(text: str) -> str:
    """
    Safely extract JSON from a triple-backtick code block if present.

    Example of expected pattern:
        ```json
        {"key": "value"}
        ```
    """
    try:
        if "```json" in text:
            # split on the first occurrence
            text = text.split("```json", 1)[1]
            # now split on the next ```
            text = text.split("```", 1)[0]
            return text.strip()
        return text
    except (IndexError, ValueError) as e:
        raise ValueError(f"Unable to extract JSON code block from: {text[:100]}") from e


def _parse_dict(text: str):
    """
    A custom fallback parser that tries to parse after extracting a JSON code block.
    This uses eval, so be cautious with untrusted content.
    """
    try:
        content = _extract_json_code_block(text)
        return eval(content)
    except Exception as e:
        raise ValueError(f"Unable to parse dictionary from text: {text[:100]}") from e


def parse_response_as_json(
    chat_session, raw_response: str, attempt: int = 0, max_attempts: int = 1
):
    """
    Attempts to parse a response string as JSON using multiple methods in turn:
    1. Direct `json.loads()` after cleaning out Markdown code fences.
    2. `ast.literal_eval()` after removing Python code fences.
    3. Fallback to custom `_parse_dict()` which uses `eval`.

    Raises a ValueError if all attempts fail.
    """
    errors = []

    # -- Method 1: Direct JSON load after removing markdown code fences
    try:
        cleaned = raw_response.replace("```json\n", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        errors.append(f"JSON load failed: {e}")

    # -- Method 2: Use ast.literal_eval with Python code fence cleanup
    try:
        cleaned = raw_response.replace("```python\n", "").replace("```", "").strip()
        # literal_eval can fail on trailing commas, single quotes, etc.
        return ast.literal_eval(cleaned)
    except (SyntaxError, ValueError) as e:
        errors.append(f"ast.literal_eval failed: {e}")

    # -- Method 3: Fallback to our custom parser that extracts a JSON code block
    try:
        return _parse_dict(raw_response)
    except Exception as e:
        errors.append(f"_parse_dict failed: {e}")

    # -- If all parsing methods fail, raise an error with aggregated messages.

    # try:
    #     if attempt < max_attempts:
    #         print(raw_response)
    #         chat_session.send_message(
    #             "Unable to parse response using any method. Identify the issue and rewrite in in ```json block"
    #         )
    #         return parse_response_as_json(chat_session, raw_response, attempt + 1)
    # except Exception as e:
    #     errors.append(f"Unable to parse response using any method: {e}")

    error_msgs = "; ".join(errors)
    raise ValueError(f"Unable to parse response using any method: {error_msgs}")


def fallback_json_parse(session, error_message):
    pass


def get_gemini_response(
    chat_session: ChatSession,
    input_msg: str,
    delay_between_retries: int = 15,
    max_retries: int = 20,
    use_cache: bool = True,
    parse_as_json: bool = True,
    max_parse_failures: int = 1,
):
    """
    Gets a response from a ChatSession, optionally caching results and parsing JSON.
    """
    if not isinstance(chat_session, ChatSession):
        raise TypeError(f"Unexpected ChatSession, got {type(chat_session).__name__}")
    if not isinstance(input_msg, str):
        raise TypeError("input_msg must be a string")

    # -- Compute cache identifier based on session history, config, etc.
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

    # -- If cache file exists and caching is enabled, load from cache
    if os.path.exists(cache_file) and use_cache:
        response_text = load_json_or_pickle(cache_file)
        return response_text

    parsed_failures = 0
    for attempt in range(1, max_retries + 1):
        try:
            # Configure a random key and send the request
            key = get_random_key()
            genai.configure(api_key=key)
            cache_text_file = cache_file.replace(".pkl", ".txt")
            if os.path.exists(cache_text_file):
                with open(cache_text_file, "r") as f:
                    response_text = f.read()
            else:
                response_text = chat_session.send_message(input_msg).text
                mkdir_or_exist(os.path.dirname(cache_text_file))
                with open(cache_text_file, "w") as f:
                    f.write(response_text)

            # Optionally parse the response as JSON
            if parse_as_json:
                response_text = parse_response_as_json(chat_session, response_text)

            # Cache if requested
            if use_cache:
                dump_json_or_pickle(response_text, cache_file)
                os.remove(cache_text_file)

            return response_text

        except Exception as e:
            # -- Final attempt: re-raise the last error
            if attempt == max_retries:
                logger.error(f"[gemini] Final attempt failed with error: {e}")
                raise

            error_str = str(e)

            # -- Check for rate limiting or server errors
            if "429" in error_str or "500" in error_str:
                # if attempt > max_retries // 2:
                logger.warning(
                    f"[gemini] {attempt}/{max_retries } -Key: {key[-10:]}- Rate limit/server error, retrying in {delay_between_retries}s"
                )
                time.sleep(delay_between_retries)
                continue

            # -- Check for JSON parse failures
            if "Unable to parse" in error_str:
                logger.error(
                    f"[gemini-JSON parse error] attempt {attempt}/{max_retries} -  {error_str}"
                )

                raise

            # -- For other errors, log and raise immediately
            logger.error(
                f"[gemini] {attempt}/{max_retries} - Unexpected error: {error_str}\nRemove this key: {key}"
            )
            raise
