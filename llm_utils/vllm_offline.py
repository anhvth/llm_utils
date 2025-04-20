import argparse
import json
from typing import List, Dict

def build_request(input_data: Dict, model_path: str, max_completion_tokens: int, output_file: str) -> List[Dict]:
    """
    Build a list of requests for the OpenAI API.

    Args:
        input_data (Dict): Dictionary containing at least "messages".
        model_path (str): Path to the model to be used.
        max_completion_tokens (int): Maximum number of tokens for the completion.
        output_file (str): Path to the output file.

    Returns:
        List[Dict]: List of request dictionaries.
    """
    messages = input_data.get("messages", [])
    requests = []
    for i, message in enumerate(messages):
        request = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_path,
                "messages": message,
                "max_completion_tokens": max_completion_tokens
            }
        }
        requests.append(request)
    with open(output_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    return requests

def run_cmd(command: str):
    """
    Execute a shell command.

    Args:
        command (str): The command to execute.
    """
    import subprocess
    subprocess.run(command, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Build requests for the OpenAI API.")
    parser.add_argument(
        '--messages_file', 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the input data with 'messages'."
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help="Path to the model to be used."
    )
    parser.add_argument(
        '--max_completion_tokens', 
        type=int, 
        required=True, 
        help="Maximum number of tokens for the completion."
    )

    args = parser.parse_args()

    # Load input data from the file
    with open(args.messages_file, 'r') as f:
        input_data = json.load(f)

    # Temporary output file for requests
    tmp_output_file = '/tmp/request.jsonl'

    # Build requests
    build_request(input_data, args.model_path, args.max_completion_tokens, tmp_output_file)

    # Output file for model results
    output_file = args.messages_file.replace('.json', f'.{args.model_path.split("/")[-1]}.json')

    # Run the model and save the output
    run_cmd(
        f"python -m vllm.entrypoints.openai.run_batch -i {tmp_output_file} -o {output_file} --model {args.model_path}"
    )

if __name__ == "__main__":
    main()
