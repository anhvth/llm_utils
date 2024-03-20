from speedy import load_by_ext
from llm_data_utils.chat_format import identify_format, transform_messages, display_chat_messages_as_html

def load_chat_dataset(path, current_format='auto', return_format='train_item'):
    assert return_format in ['train_item', 'chat_lm'], "The return format is not recognized. Please specify the return format."
    data= load_by_ext(path)
    assert isinstance(data, list), "The data is not in the correct format. Please check the format of the data."
    if current_format == 'auto':
        current_format = identify_format(data[0])
    if current_format != return_format:
        from loguru import logger
        logger.info(f"Converting the {path} from {current_format} to {return_format}.")
        items = [transform_messages(item, current_format, return_format) for item in data]
    else:
        items = data

    return items


if __name__ == '__main__':
    train_items = load_chat_dataset('/anhvth5/data/chat-formated-dataset/legal/CCO_context_conversation_orchestration.json')
    
    
    html = display_chat_messages_as_html(train_items[0])
    with open('test.html', 'w') as f:
        f.write(html)