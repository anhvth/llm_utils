#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
USAGE:
Serve models and LoRAs with vLLM:

Serve a base model:
svllm serve --model MODEL_NAME --gpus GPU_GROUPS --host_port host:port

Add a LoRA to a served model:
svllm add_lora --lora LORA_NAME LORA_PATH --host_port host:port

Unload a LoRA from a served model:
svllm unload_lora --lora LORA_NAME --host_port host:port

List models served on a specific host:port:
svllm list_models --host_port host:port

"""
from glob import glob
import os
import subprocess
import time
from typing import List, Literal, Optional
# from fastcore.script import call_parse # Removed as argparse is used directly
from loguru import logger
import argparse
import requests
import openai
from pyfzf.pyfzf import FzfPrompt # Added import for pyfzf
from speedy_utils import jloads, load_by_ext, memoize # Assuming speedy_utils is available
LORA_DIR = os.environ.get("LORA_DIR", "/loras")
LORA_DIR = os.path.abspath(LORA_DIR)
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
logger.info(f"LORA_DIR: {LORA_DIR}")


def get_vllm():
    """Finds the vLLM binary path."""
    try:
        vllm_binary_path = subprocess.check_output("which vllm", shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        vllm_binary_path = None # Handle case where 'which' fails

    VLLM_BINARY = os.getenv("VLLM_BINARY", vllm_binary_path)
    if not VLLM_BINARY or not os.path.exists(VLLM_BINARY):
         raise FileNotFoundError(
             f"vLLM binary not found. Tried path: {VLLM_BINARY}. "
             "Ensure vLLM is installed and in PATH, or set the VLLM_BINARY environment variable."
         )
    logger.info(f"Using vLLM binary: {VLLM_BINARY}")
    return VLLM_BINARY


def model_list(host_port, api_key='abc'):
    """Lists models available on the specified vLLM server."""
    try:
        client = openai.OpenAI(base_url=f"http://{host_port}/v1", api_key=api_key)
        models = client.models.list()
        if not models.data:
            print(f"No models found on http://{host_port}/v1")
            return
        print(f"Models available on http://{host_port}/v1:")
        for model in models:
            print(f"  - Model ID: {model.id}")
    except openai.APIConnectionError as e:
        logger.error(f"Could not connect to server at http://{host_port}/v1: {e}")
    except Exception as e:
        logger.error(f"An error occurred while listing models: {e}")



def add_lora(
    lora_name_or_path: str,
    host_port: str, # Made host_port required and positional for clarity
    served_model_name: str,
    url: str = "http://{host_port}/v1/load_lora_adapter", # Updated URL structure? Check vLLM docs
) -> dict:
    """Adds a LoRA adapter to a running vLLM server."""
    url = url.format(host_port=host_port) # Use format for clarity
    headers = {"Content-Type": "application/json"}

    lora_absolute_path = os.path.abspath(lora_name_or_path)
    if not os.path.exists(lora_absolute_path):
         logger.error(f"LoRA path does not exist: {lora_absolute_path}")
         return {"error": f"LoRA path does not exist: {lora_absolute_path}"}

    # Construct the payload based on typical vLLM API (verify this)
    data = {
             "lora_name": served_model_name, # The name to refer to this LoRA by
             "lora_path": lora_absolute_path # Path on the server machine
    }

    # Include lora_module if provided and supported by the endpoint
    # This part might need adjustment depending on how vLLM handles target modules now
    # if lora_module:
    #     data["lora_module"] = lora_module # Or nested within lora_request?

    logger.info(f"Attempting to load LoRA adapter:")
    logger.info(f"  URL: {url}")
    logger.info(f"  Headers: {headers}")
    logger.info(f"  Data: {data}")

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        logger.success(f"Successfully requested loading of LoRA adapter '{served_model_name}' from path '{lora_absolute_path}' on {host_port}.")

        # Handle potential non-JSON success responses (e.g., simple "OK" text)
        try:
            return response.json()
        except ValueError: # Catches JSONDecodeError
            response_text = response.text.strip()
            if response.status_code == 200 and not response_text:
                 return {"status": "success", "message": "Request successful, empty response body."}
            elif response.status_code == 200:
                 return {"status": "success", "message": response_text}
            else:
                 # Should be caught by raise_for_status, but as a fallback
                 return {"status": "unknown", "message": f"Non-JSON response (Status {response.status_code}): {response_text}"}


    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: Could not connect to the server at {url}. Is it running? Details: {e}")
        return {"error": f"Connection failed: {e}"}
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out while trying to reach {url}: {e}")
        return {"error": f"Request timed out: {e}"}
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} {e.response.reason}. Response: {e.response.text}")
        return {"error": f"HTTP error: {e.response.status_code}", "response": e.response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected request error occurred: {e}")
        return {"error": f"Request failed: {e}"}
    except Exception as e:
         logger.error(f"An unexpected error occurred in add_lora: {e}")
         return {"error": f"Unexpected error: {str(e)}"}


def unload_lora(lora_name: str, host_port: str):
    """Unloads a LoRA adapter from a running vLLM server."""
    # Note: Verify the correct endpoint URL and payload structure with vLLM documentation.
    # Assuming an endpoint like /v1/lora_adapters/unload
    url = f"http://{host_port}/v1/unload_lora_adapter"
    headers = {"Content-Type": "application/json"}
    data = {
         "lora_request": {
             "lora_name": lora_name
         }
         # Or maybe just {"lora_name": lora_name} - check docs
    }

    logger.info(f"Attempting to unload LoRA adapter '{lora_name}' from {host_port}")
    logger.info(f"  URL: {url}")
    logger.info(f"  Data: {data}")

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.success(f"Successfully requested unloading of LoRA adapter: {lora_name}")
        try:
            return response.json()
        except ValueError:
            return {"status": "success", "message": response.text or "Unload request successful."}

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: Could not connect to the server at {url}. Is it running? Details: {e}")
        return {"error": f"Connection failed: {e}"}
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} {e.response.reason}. Failed to unload LoRA '{lora_name}'. Response: {e.response.text}")
        return {"error": f"HTTP error: {e.response.status_code}", "response": e.response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed while unloading LoRA '{lora_name}': {e}")
        return {"error": f"Request failed: {e}"}
    except Exception as e:
         logger.error(f"An unexpected error occurred in unload_lora: {e}")
         return {"error": f"Unexpected error: {str(e)}"}


@memoize # Assuming memoize works as intended for caching
def fetch_chat_template(template_name: str = 'qwen') -> str:
    """
    Fetches a chat template file from a remote repository or local cache.

    Args:
        template_name (str): Name of the chat template. Defaults to 'qwen'.

    Returns:
        str: Path to the downloaded or cached chat template file.

    Raises:
        AssertionError: If the template_name is not supported.
        ValueError: If the file URL is invalid or fetching fails.
        ImportError: If 'requests' library is not installed.
    """
    # Define supported templates within the function or load from config
    supported_templates = [
        'alpaca', 'chatml', 'gemma-it', 'llama-2-chat',
        'mistral-instruct', 'qwen2.5-instruct', 'saiga',
        'vicuna', 'qwen'
    ]

    # Map 'qwen' alias if needed
    effective_template_name = 'qwen2.5-instruct' if template_name == 'qwen' else template_name

    if effective_template_name not in supported_templates:
        raise AssertionError(
            f"Chat template '{template_name}' (resolved to '{effective_template_name}') not supported. "
            f"Please choose from {supported_templates}."
        )

    remote_url = (
        f'https://raw.githubusercontent.com/chujiezheng/chat_templates/'
        f'main/chat_templates/{effective_template_name}.jinja'
    )
    local_cache_path = f'/tmp/chat_template_{effective_template_name}.jinja' # Use effective name

    # Check cache first
    if os.path.exists(local_cache_path):
         logger.info(f"Using cached chat template: {local_cache_path}")
         return local_cache_path

    logger.info(f"Fetching chat template '{effective_template_name}' from {remote_url}...")

    try:
        import requests # Import locally to make dependency optional if not using chat templates
    except ImportError:
        raise ImportError("The 'requests' library is required to fetch chat templates. Please install it (`pip install requests`).")

    try:
        response = requests.get(remote_url, timeout=10) # Added timeout
        response.raise_for_status() # Check for HTTP errors

        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True) # Ensure /tmp exists
        with open(local_cache_path, 'w', encoding='utf-8') as file: # Specify encoding
            file.write(response.text)

        logger.success(f"Successfully fetched and cached chat template at: {local_cache_path}")
        return local_cache_path

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch chat template from {remote_url}: {e}")
    except IOError as e:
         raise ValueError(f"Failed to write chat template to cache {local_cache_path}: {e}")


def get_chat_template(template_name: str) -> str:
    """Wrapper to fetch the chat template, handling potential errors."""
    try:
        return fetch_chat_template(template_name)
    except (AssertionError, ValueError, ImportError) as e:
        logger.error(f"Error getting chat template '{template_name}': {e}")
        # Decide how to handle: raise error, return None, or exit
        raise  # Re-raise the exception to stop execution if template is critical


def serve(
    model: str,
    gpu_groups: str,
    host: str, # Added host parameter
    port_start: int, # Renamed from host_port to avoid confusion
    served_model_name: Optional[str] = None,
    gpu_memory_utilization: float = 0.93,
    dtype: str = "auto", # Changed default to auto
    max_model_len: int = 8192,
    enable_lora: bool = False,
    is_bnb: bool = False,
    eager: bool = False,
    chat_template: Optional[str] = None,
    lora_modules: Optional[List[str]] = None,
    pipeline_parallel: int = 1, # Added pipeline parallel arg
    # extra_args: Optional[List[str]] = None # For future flexibility
):
    """Starts vLLM server instances based on the provided configuration."""
    logger.info("Starting vLLM server instance(s)...")

    gpu_groups_list = gpu_groups.split(";") # Allow semi-colon for multiple groups
    num_instances = len(gpu_groups_list)
    logger.info(f"Found {num_instances} GPU group(s): {gpu_groups_list}")

    try:
        VLLM_BINARY = get_vllm()
    except FileNotFoundError as e:
        logger.error(e)
        return # Exit if vllm binary isn't found

    env_prefix = ""
    if enable_lora:
        # Note: VLLM_ALLOW_RUNTIME_LORA_UPDATING might be deprecated or changed.
        # Modern vLLM handles this via --enable-lora implicitly? Verify vLLM docs.
        # os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True" # Set in process env if needed
        env_prefix = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True " # Prepend to command instead
        logger.info("Runtime LoRA updating enabled via environment variable (verify if still needed).")

    # Auto-detect quantization if not explicitly set
    if not is_bnb and model and ("bnb" in model.lower() or "4bit" in model.lower()):
        is_bnb = True
        logger.info(f"Auto-detected quantization required for model: {model}. Enabling BNB.")


    for i, gpu_group in enumerate(gpu_groups_list):
        port = port_start + i
        # Validate GPU IDs in the group
        try:
             gpu_ids = [int(g.strip()) for g in gpu_group.split(",") if g.strip()]
             if not gpu_ids: raise ValueError("Empty GPU group.")
             gpu_group_str = ",".join(map(str, gpu_ids))
        except ValueError as e:
             logger.error(f"Invalid GPU group format '{gpu_group}'. Must be comma-separated integers. Error: {e}")
             continue # Skip this instance

        tensor_parallel = len(gpu_ids)
        current_host = host # Use the provided host

        # Construct the command
        cmd_env = {**os.environ} # Inherit environment
        if HF_HOME:
            cmd_env['HF_HOME'] = HF_HOME
        cmd_env['CUDA_VISIBLE_DEVICES'] = gpu_group_str

        # Base command parts
        cmd_list = [
            VLLM_BINARY,
            "serve",
            model,
            "--host", current_host,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel), # Corrected argument name
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--dtype", dtype,
            "--max-model-len", str(max_model_len),
            "--disable-log-requests", # Good default for cleaner logs
            # Consider adding --disable-log-stats if not needed
            "--trust-remote-code", # Often necessary for custom models
        ]

        # Conditional flags
        if served_model_name:
            cmd_list.extend(["--served-model-name", served_model_name])
        else:
            # Default served_model_name might be useful, e.g., derived from model path
            pass

        if is_bnb:
            # vLLM might auto-detect quantization needs based on model files.
            # Explicit flags might be for specific formats. Check docs.
            # Common options: --quantization awq, gptq, squeezellm, marlin, None (fp16/bf16)
            # BNB integration might be handled differently (e.g. needing bitsandbytes installed)
            # Assuming direct flag for now, VERIFY THIS with current vLLM:
             cmd_list.extend(["--quantization", "bitsandbytes"]) # This flag might not exist, BNB might be implicit
             logger.warning("Using '--quantization bitsandbytes'. Verify this flag is correct for your vLLM version.")
             # You might not need --load-format bitsandbytes anymore.

        if enable_lora:
            cmd_list.append("--enable-lora")
            # cmd_list.append("--fully-sharded-loras") # Consider if needed/available

        if eager:
            cmd_list.append("--enforce-eager")

        if pipeline_parallel > 1:
            cmd_list.extend(["--pipeline-parallel-size", str(pipeline_parallel)])


        if chat_template:
            try:
                template_file_path = get_chat_template(chat_template)
                cmd_list.extend(["--chat-template", template_file_path])
            except Exception as e:
                logger.error(f"Failed to get chat template '{chat_template}', proceeding without it. Error: {e}")
                # Decide if you want to stop here or continue without the template

        if lora_modules:
            # Format: --lora-modules NAME=PATH [NAME=PATH ...]
            lora_module_args = []
            if len(lora_modules) % 2 != 0:
                logger.error(f"Invalid lora_modules list: {lora_modules}. Must have an even number of elements (name path name path...).")
            else:
                for j in range(0, len(lora_modules), 2):
                    name = lora_modules[j]
                    path = os.path.abspath(lora_modules[j + 1]) # Ensure absolute path
                    if not os.path.exists(path):
                        logger.warning(f"LoRA path does not exist for module '{name}': {path}. Skipping.")
                        continue
                    lora_module_args.append(f"{name}={path}")

                if lora_module_args:
                    cmd_list.append("--lora-modules")
                    cmd_list.extend(lora_module_args) # Add multiple args correctly
                else:
                    logger.warning("No valid LoRA modules found to pass to --lora-modules.")


        # Logging setup for the instance
        log_dir = "/tmp/vllm_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"vllm_{current_host}_{port}.log")

        final_cmd_str = " ".join(cmd_list)
        # Add environment prefix if needed
        if env_prefix:
            final_cmd_str = env_prefix + final_cmd_str

        # Tmux command to run in background and log output
        # Ensure tmux session names are unique if running multiple scripts
        tmux_session_name = f"vllm_{current_host.replace('.', '_')}_{port}"

        # Escape the command properly for bash inside tmux
        escaped_cmd = final_cmd_str.replace("'", "'\\''") # Basic escaping for single quotes

        tmux_cmd = (
            f"tmux new-session -d -s {tmux_session_name} "
            f"'bash -c \"{escaped_cmd} 2>&1 | tee {log_file}\"'"
        )

        logger.info(f"🚀 Launching vLLM instance {i+1}/{num_instances} in tmux session '{tmux_session_name}':")
        logger.info(f"   Host: {current_host}, Port: {port}, GPUs: {gpu_group_str}")
        logger.info(f"   Log File: {log_file}")
        # logger.debug(f"   Full Command: {final_cmd_str}") # Log the command if needed
        logger.debug(f"   Tmux Command: {tmux_cmd}")

        try:
            subprocess.run(tmux_cmd, shell=True, check=True)
            logger.success(f"   Instance launched successfully.")
            # Optional: Add a small delay or health check here
            time.sleep(2) # Small delay before launching next one
        except subprocess.CalledProcessError as e:
            logger.error(f"   Failed to launch instance in tmux session '{tmux_session_name}'. Error: {e}")
            logger.error(f"   Failed command: {tmux_cmd}")
        except Exception as e:
             logger.error(f"   An unexpected error occurred launching instance in tmux: {e}")


def get_args():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Manage vLLM Server Instances (Serve, Add/Unload LoRA, List Models)",
        formatter_class=argparse.RawTextHelpFormatter, # Preserve formatting in help
        epilog="""\
Examples:
  # Serve a base model on GPUs 0,1 using port 8000
  svllm serve --model HuggingFaceH4/zephyr-7b-beta --gpus 0,1 --host_port localhost:8000

  # Serve a model with quantization enabled on GPUs 2,3 starting at port 8001
  svllm serve --model TheBloke/Mistral-7B-Instruct-v0.1-GPTQ --gpus 2,3 --host_port localhost:8001 --quantization gptq

  # Serve a base model and preload LoRA adapters (requires --enable-lora)
  svllm serve --model mistralai/Mistral-7B-v0.1 --gpus 0 --host_port localhost:8000 --enable-lora \\
              --lora_modules my_lora1 /path/to/lora1 other_lora /path/to/lora2

  # Serve a LoRA directly (infers base model from adapter_config.json) on GPUs 0,1 port 8000
  svllm serve --lora my_cool_lora /path/to/my_cool_lora --gpus 0,1 --host_port localhost:8000

  # Add a LoRA adapter named 'new_adapter' to the server at localhost:8000
  svllm add_lora --lora new_adapter /path/to/new_adapter --host_port localhost:8000

  # Unload the LoRA adapter named 'my_lora1' from localhost:8000
  svllm unload_lora --lora my_lora1 --host_port localhost:8000

  # List models/adapters available on the server at localhost:8000
  svllm list_models --host_port localhost:8000

"""
    )

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode to run the script in")

    # --- Serve Mode ---
    parser_serve = subparsers.add_parser("serve", help="Start vLLM server instance(s)")
    parser_serve.add_argument("--model", "-m", type=str, help="Full name or path of the base model to serve (required unless --lora is used)")
    parser_serve.add_argument("--gpus", "-g", type=str, required=True, help="Comma-separated GPU IDs for one instance, or semi-colon separated groups for multiple instances (e.g., '0,1;2,3')")
    parser_serve.add_argument("--host_port", '-hp', type=str, required=True, help="Host and starting port for the server(s) in format: 'host:port' (e.g., 'localhost:8000', '0.0.0.0:8155')")
    parser_serve.add_argument("--lora", "-l", nargs=2, metavar=("LORA_NAME", "LORA_PATH"), help="If specified, serve this LoRA. The base model is inferred from adapter_config.json. Overrides --model if both are given.")
    parser_serve.add_argument("--served_model_name", type=str, help="Custom name for the served model endpoint (defaults based on model name)")
    parser_serve.add_argument("--gpu_memory_utilization", '-gmu', type=float, default=0.93, help="GPU memory utilization fraction (0.0 to 1.0, default: 0.93)")
    parser_serve.add_argument("--dtype", type=str, default="auto", help="Data type (e.g., 'auto', 'bfloat16', 'float16', 'float32', default: auto)")
    parser_serve.add_argument("--max_model_len", '-mml', type=int, default=None, help="Maximum model context length (default: auto-detect)")
    parser_serve.add_argument("--enable-lora", action="store_true", help="Enable LoRA support in the vLLM server")
    # Updated quantization argument - check vLLM docs for current accepted values
    parser_serve.add_argument("--quantization", "-q", type=str, default=None, choices=['awq', 'gptq', 'squeezellm', 'marlin'], help="Quantization method (e.g., awq, gptq). BNB might be handled implicitly/differently.")
    parser_serve.add_argument("--bnb", action="store_true", help="[Deprecated - Use --quantization or rely on auto-detection] Explicitly enable bitsandbytes (may not be needed)")
    parser_serve.add_argument("--eager", action="store_true", help="Enable eager execution mode")
    parser_serve.add_argument("--chat_template", type=str, help="Name of the chat template to load (e.g., 'qwen', 'llama-2-chat') or path to a Jinja template file.")
    parser_serve.add_argument("--lora_modules", "-lm", nargs="+", type=str, help="Preload LoRA modules at startup. Format: name1 path1 [name2 path2 ...]. Requires --enable-lora.")
    parser_serve.add_argument("--pipeline_parallel", "-pp", default=1, type=int, help="Number of pipeline parallel stages")
    # parser_serve.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional arguments to pass directly to vllm serve") # Example for future

    # --- Add LoRA Mode ---
    parser_add_lora = subparsers.add_parser("add_lora", help="Add a LoRA adapter to a running server")
    parser_add_lora.add_argument("--lora", "-l", nargs=2, metavar=("LORA_NAME", "LORA_PATH"), required=True, help="Name and path of the LoRA adapter to add")
    parser_add_lora.add_argument("--host_port", '-hp', type=str, required=True, help="Host and port of the target vLLM server (host:port)")
    # parser_add_lora.add_argument("--lora_module", type=str, help="Optional: Target module for LoRA (depends on vLLM API)") # If needed

    # --- Unload LoRA Mode ---
    parser_unload_lora = subparsers.add_parser("unload_lora", help="Unload a LoRA adapter from a running server")
    # Allow specifying lora by name only for unload
    parser_unload_lora.add_argument("--lora", "-l", type=str, required=True, metavar="LORA_NAME", help="Name of the LoRA adapter to unload")
    parser_unload_lora.add_argument("--host_port", '-hp', type=str, required=True, help="Host and port of the target vLLM server (host:port)")

    # --- List Models Mode ---
    parser_list = subparsers.add_parser("list_models", help="List models/adapters available on a running server")
    parser_list.add_argument("--host_port", '-hp', type=str, required=True, help="Host and port of the target vLLM server (host:port)")
    parser_list.add_argument("--api_key", type=str, default="abc", help="API key for the server (if required, default: abc)")

    # --- Global Arguments (example, not used here) ---
    # parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging") # Could be added

    return parser.parse_args()

def parse_host_port(host_port_str: str) -> tuple[str, int]:
     """Parses 'host:port' string into (host, port)."""
     try:
         host, port_str = host_port_str.rsplit(':', 1)
         port = int(port_str)
         if not host: # Handle cases like ":8000" -> use default host
             host = "localhost"
         return host, port
     except (ValueError, AttributeError):
         raise argparse.ArgumentTypeError(f"Invalid host:port format: '{host_port_str}'. Expected 'host:port' (e.g., 'localhost:8000').")


def main():
    """Main entry point for the script."""
    args = get_args()

    # --- Mode Dispatch ---
    if args.mode == "serve":
        # --- Serve Logic ---
        try:
            host, port_start = parse_host_port(args.host_port)
        except argparse.ArgumentTypeError as e:
             logger.error(e)
             return # Exit on invalid host:port

        model_to_serve = args.model
        lora_modules_to_preload = args.lora_modules
        enable_lora_flag = args.enable_lora
        is_bnb_flag = args.bnb # Note deprecation warning
        quantization_flag = args.quantization # Preferred way

        # Handle --lora argument for serving a LoRA directly
        if args.lora:
            lora_name, lora_path = args.lora
            lora_path = os.path.abspath(lora_path)
            logger.info(f"Serving LoRA directly: Name='{lora_name}', Path='{lora_path}'")
            enable_lora_flag = True # Serving a LoRA requires lora enabled

            # Infer base model from adapter_config.json
            lora_config_path = os.path.join(lora_path, "adapter_config.json")
            if os.path.exists(lora_config_path):
                try:
                    config = load_by_ext(lora_config_path) # Assumes load_by_ext handles JSON
                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                         logger.info(f"Inferred base model from config: {base_model}")
                         # Override args.model if --lora is used
                         model_to_serve = base_model
                         # Auto-detect BNB requirement from base model name in config?
                         if not quantization_flag and ("bnb" in base_model.lower() or "4bit" in base_model.lower()):
                              logger.info(f"Auto-detecting quantization needed based on inferred base model name.")
                              is_bnb_flag = True # Or set appropriate quantization_flag if possible
                    else:
                        logger.error(f"'base_model_name_or_path' not found in {lora_config_path}. Cannot infer base model.")
                        return # Cannot proceed without a base model
                except Exception as e:
                     logger.error(f"Error reading LoRA config {lora_config_path}: {e}")
                     return
            else:
                logger.error(f"LoRA config file not found: {lora_config_path}. Cannot infer base model.")
                return # Cannot proceed

            # Add the specified LoRA to the preload list if not already there via --lora_modules
            if not lora_modules_to_preload:
                lora_modules_to_preload = [lora_name, lora_path]
            else:
                 # Avoid duplicates if specified both ways
                 is_already_listed = False
                 for i in range(0, len(lora_modules_to_preload), 2):
                     if lora_modules_to_preload[i] == lora_name:
                         is_already_listed = True
                         # Optionally update path?
                         lora_modules_to_preload[i+1] = lora_path
                         break
                 if not is_already_listed:
                     lora_modules_to_preload.extend([lora_name, lora_path])
            logger.info(f"LoRA modules to preload: {lora_modules_to_preload}")

        # Check if a model is defined either via --model or inferred from --lora
        if not model_to_serve:
            logger.error("Model not specified. Use --model <model_name> or --lora <name> <path>.")
            return

        # Warning if old --bnb flag is used with new --quantization
        if is_bnb_flag and quantization_flag:
            logger.warning("Both --bnb and --quantization are specified. Prefer using only --quantization.")
            # Decide precedence or rely on vLLM's handling
        elif is_bnb_flag:
             logger.warning("Using deprecated --bnb flag. Consider using --quantization if applicable, or rely on auto-detection.")
             # You might want to map --bnb to a specific --quantization value if possible,
             # otherwise pass it along as 'is_bnb_flag' to the serve function for now.

        serve(
            model=model_to_serve,
            gpu_groups=args.gpus,
            host=host,
            port_start=port_start,
            served_model_name=args.served_model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len if args.max_model_len is not None else 8192, # Handle None default properly
            enable_lora=enable_lora_flag,
            is_bnb=is_bnb_flag, # Pass the potentially deprecated flag for now
            quantization=quantization_flag, # Pass the preferred flag
            eager=args.eager,
            chat_template=args.chat_template,
            lora_modules=lora_modules_to_preload,
            pipeline_parallel=args.pipeline_parallel,
        )


    elif args.mode == "add_lora":
        # --- Add LoRA Logic ---
        try:
            host, port = parse_host_port(args.host_port)
            target_host_port = f"{host}:{port}" # Reconstruct for functions expecting host:port string
        except argparse.ArgumentTypeError as e:
             logger.error(e)
             return

        lora_name, lora_path = args.lora
        result = add_lora(
            lora_name_or_path=lora_path,
            host_port=target_host_port,
            served_model_name=lora_name, # Name to use for this LoRA on the server
            # lora_module=args.lora_module # If needed
        )
        logger.info(f"Add LoRA result: {result}")


    elif args.mode == "unload_lora":
         # --- Unload LoRA Logic ---
        try:
            host, port = parse_host_port(args.host_port)
            target_host_port = f"{host}:{port}"
        except argparse.ArgumentTypeError as e:
             logger.error(e)
             return

        lora_name = args.lora # Unload uses only the name
        result = unload_lora(lora_name, host_port=target_host_port)
        logger.info(f"Unload LoRA result: {result}")


    elif args.mode == "list_models":
        # --- List Models Logic ---
        try:
            host, port = parse_host_port(args.host_port)
            target_host_port = f"{host}:{port}"
        except argparse.ArgumentTypeError as e:
             logger.error(e)
             return

        model_list(target_host_port, api_key=args.api_key)

    else:
        # This should not happen due to subparsers(required=True)
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    # Setup logger (optional: configure level based on args.verbose if added)
    # logger.add("file_{time}.log") # Example file logging
    main()