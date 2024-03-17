from IPython.core.display import HTML, Markdown, display

def _convert_train_item_to_messages(item, default_system_message="You are a helpful assistant.", print_msg=False):
    if isinstance(item, list):
        return [_convert_train_item_to_messages(item) for item in item]

    messages = []
    system_msg = item.get("system", '')
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    elif default_system_message:
        messages.append({"role": "system", "content": default_system_message})
    conversations = item.get("conversations", [])
    assert conversations, "The item does not have any conversations."
    for conversation in item.get("conversations", []):
        role = conversation['from']
        content = conversation['value']
        messages.append({"role": role, "content": content})

    return messages

def to_chatlm_messages(messages, input_format='auto', log=False, assistant_prefix='', print_msg=False):
    if input_format == 'auto':
        if isinstance(messages, list):
            input_format = 'openai_chatlm'
            assert messages[0].get("role") is not None, "The input format is not recognized. Please specify the input format."
        elif isinstance(messages, dict):
            messages = _convert_train_item_to_messages(messages)
            input_format = 'train_item'
        elif isinstance(messages, str):
            # assume it has format <|im_start|>role\n content<|im_end|> use regex to parse
            assert '<|im_end|>' in messages, "The input format is not recognized. Please specify the input format."
            input_format = 'openai_chatlm'
            # 1. split by <|im_end|>
            parts = messages.split('<|im_end|>')
            # for each part, split by <|im_start|> to get role and content
            messages = []
            for part in parts:
                if not part.strip():
                    continue
                # role is after <|im_start|> and before \n
                role = part.split('<|im_start|>')[1].split('\n')[0]
                # content is after first \n
                content = part.split('\n', 1)[1]
                messages.append({"role": role.strip(), "content": content.strip()})
    
        
    return messages

def convert_to_prompt(messages, input_format='auto', log=False, assistant_prefix='', print_msg=False):
    if messages[-1]["role"] == "assistant":
        last_message = messages.pop()
    else:
        last_message = None

    # Convert to prompt
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"<|im_start|>{role}\n {content}<|im_end|>\n"
    prompt += '<|im_start|>assistant\n'+assistant_prefix
    return prompt, last_message['content'] if last_message else None




def display_messages(messages, max_messages=10):
    messages = to_chatlm_messages(messages)
    num_trimmed = len(messages) - max_messages

    html_content = '<div style="font-family: Arial, sans-serif;">'
    if num_trimmed > 0:
        html_content += f'<p style="color:#FFAAAA;">({num_trimmed} messages trimmed...)</p>'
    for message in messages[-max_messages:]:
        if message['role'] == 'system':
            color = '#FFAAAA'  # Light red
        elif message['role'] == 'user':
            color = '#AAFFAA'  # Light green
        elif message['role'] == 'assistant':
            color = '#AAAAFF'  # Light blue
        elif message['role'] == 'function':
            # yellow
            color = '#AFFFFF'
        else:
            color = '#FFFFFF'  # White, just in case there's an unexpected role

        content_safe = message['content'].replace('<', '&lt;').replace('>', '&gt;')
        # handle \n
        content_safe = content_safe.replace('\n', '<br>')
        
        html_content += f'<div style="background-color:{color}; margin:10px; padding:10px; border-radius:8px;">'
        html_content += f'<strong>{message["role"].capitalize()}:</strong> <span>{content_safe}</span>'
        html_content += '</div>'
    html_content += '</div>'
    display(HTML(html_content))