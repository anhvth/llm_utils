from IPython.core.display import HTML, Markdown, display


def identify_format(item):
    if isinstance(item, list) and 'role' in item[0]:
        return 'chat_lm'
    if isinstance(item, dict):
        if 'conversations' in item:
            return 'train_item'
    raise ValueError(f"The format of the item is not recognized. \n{type(item)=}, \n{item=}")

def transform_training_data_to_message_format(item, default_system_message="You are a helpful assistant.", print_msg=False):
    if isinstance(item, list):
        return [transform_training_data_to_message_format(item) for item in item]

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
from copy import deepcopy as deeopcopy

def transform_messages(item, frm, to):
    item = deeopcopy(item)
    if frm != to:
        # convert item to chat_lm format
        messages = normalize_messages_to_chat_format(item, input_format='auto')
        if to == 'train_item':
            if messages[0]["role"] == "system":
                system_message = messages[0]["content"]
                ret = {"conversations": [], system_message: system_message}
                for message in messages[1:]:
                    ret["conversations"].append({"from": message["role"], "value": message["content"]})
            else:
                system_message = "You are a helpful assistant."
                ret = {"conversations": [], system_message: system_message}
                for message in messages:
                    ret["conversations"].append({"from": message["role"], "value": message["content"]})
            return ret

    else:
        return item


def normalize_messages_to_chat_format(messages, input_format='auto', log=False, assistant_prefix='', print_msg=False):
    if input_format == 'auto':
        if isinstance(messages, list):
            input_format = 'chatlm'
            assert messages[0].get("role") is not None, "The input format is not recognized. Please specify the input format."
        elif isinstance(messages, dict):
            messages = transform_training_data_to_message_format(messages)
            input_format = 'train_item'
        elif isinstance(messages, str):
            # assume it has format <|im_start|>role\n content<|im_end|> use regex to parse
            assert '<|im_end|>' in messages, "The input format is not recognized. Please specify the input format."
            input_format = 'chatlm'
            # 1. split by <|im_end|>
            parts = messages.split('<|im_end|>')
            # for each part, split by <|im_start|> to get role and content
            messages = []
            for part in parts:
                if not part.strip():
                    continue
                # role is after <|im_start|> and before \n
                role = part.split('<|im_start|>')[1].split('\n')[0]
                # content is after |>role\n
                content = part.split(f'<|im_start|>{role}\n')[1]
                content = content.split('<|im_end|>')[0]

                messages.append({"role": role.strip(), "content": content.strip()})

    return messages

def create_prompt_from_messages(messages, input_format='auto', log=False, assistant_prefix='', print_msg=False):
    """
        Returns:
        - prompt: str (<|im_start|>role\n content<|im_end|>...<|im_start|>assistant\n{assistant_prefix})
        - last_message: str
    """
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




def display_chat_messages_as_html(messages, max_messages=10, theme='light', to_file='/tmp/chat.html'):
    messages = normalize_messages_to_chat_format(messages)
    num_trimmed = len(messages) - max_messages

    html_content = '<div style="font-family: Arial, sans-serif;">'
    if num_trimmed > 0:
        html_content += f'<p style="color:#FFAAAA;">({num_trimmed} messages trimmed...)</p>'
    for message in messages[-max_messages:]:

        if theme == 'light':
            color_scheme = {
                'system': {'background': '#FFAAAA', 'text': '#000000'},  # Light red background, black text
                'user': {'background': '#AAFFAA', 'text': '#000000'},  # Light green background, black text
                'assistant': {'background': '#AAAAFF', 'text': '#000000'},  # Light blue background, black text
                'function': {'background': '#AFFFFF', 'text': '#000000'},  # Light yellow background, black text
                'default': {'background': '#FFFFFF', 'text': '#000000'},  # White background, black text
            }
        else:  # For dark theme or other themes
            color_scheme = {
                'system': {'background': '#D9534F', 'text': '#FFFFFF'},  # Darker red background, white text
                'user': {'background': '#5CB85C', 'text': '#FFFFFF'},  # Darker green background, white text
                'assistant': {'background': '#5BC0DE', 'text': '#FFFFFF'},  # Darker blue background, white text
                'function': {'background': '#F0AD4E', 'text': '#FFFFFF'},  # Darker yellow background, white text
                'default': {'background': '#2C3E50', 'text': '#FFFFFF'},  # Dark slate background, white text
            }

        # Extract both background and text colors based on the role
        background_color = color_scheme.get(message['role'], color_scheme['default'])['background']
        text_color = color_scheme.get(message['role'], color_scheme['default'])['text']

        content_safe = message['content'].replace('<', '&lt;').replace('>', '&gt;')
        # handle \n
        content_safe = content_safe.replace('\n', '<br>')
                
        html_content += f'<div style="background-color:{background_color}; margin:10px; padding:10px; border-radius:8px;">'
        html_content += f'<strong style="color:{text_color};">{message["role"].capitalize()}:</strong> <span style="color:{text_color};">{content_safe}</span>'
        html_content += '</div>'

    html_content += '</div>'
    display(HTML(html_content))
    if to_file:
        with open(to_file, 'w') as f:
            f.write(html_content)
    return html_content
    
def generate_one_turn_chat_message(
    user_msg, system_msg=None, assistant_prefix=None, previous_conversation=None
):
    _mesages = []
    if system_msg:
        _mesages.append({"role": "system", "content": system_msg})
    _mesages.append({"role": "user", "content": user_msg})

    text = ''
    for turn in _mesages:
        # if turn["role"] == "user":
        turn["content"] = turn["content"].strip()
        text += '<|im_start|>{role}\n{content}<|im_end|>\n'.format(**turn)
    text += '<|im_start|>assistant\n'
    if assistant_prefix:
        text += assistant_prefix
    if previous_conversation:
        if not previous_conversation.endswith("\n"):
            previous_conversation += "\n"
        text = previous_conversation + text
    return text
