from copy import deepcopy
from typing import List, Optional

from IPython.display import HTML, display
from loguru import logger
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

    def to_html(self):
        message = self
        color_scheme = {
            'system': {'background': '#FFAAAA', 'text': '#000000'},
            'user': {'background': '#AAFFAA', 'text': '#000000'},
            'assistant': {'background': '#AAAAFF', 'text': '#000000'}
        }

        # Get colors for the current message
        background_color, text_color = color_scheme.get(message.role, {'background': '#FFFFFF', 'text': '#000000'}).values()

        # Safe message content
        content_safe = message.content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>').replace(' ', '&nbsp;')

        # HTML for the message
        html_content = ''
        html_content += f'<div style="background-color:{background_color}; margin:10px; padding:10px; border-radius:8px;">'
        html_content += f'<strong style="color:{text_color};">{message.role.capitalize()}:</strong> <span style="color:{text_color};">{content_safe}</span>'
        html_content += '</div>'
        return html_content

class Conversation(BaseModel):
    messages: List[Message] = []

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def get_messages(self) -> List[Message]:
        return self.messages

    def clear_messages(self):
        self.messages.clear()

    def to_dict(self) -> dict:
        return {
            "messages": [message.dict() for message in self.messages]
        }

    @classmethod
    def from_messages_list(cls, messages_list: List[dict]):
        messages = []
        for msg_dict in messages_list:
            try:
                msg = Message(**msg_dict) 
                messages.append(msg)
            except Exception   as e:
                logger.error(f"Error while parsing message {msg_dict}: {e}, this message will be ignored")
        return cls(messages=messages)

    def to_html(self, msg_ids:List[int]=None, theme: str = 'light', to_file: Optional[str] = None):
        messages = self.get_messages()
        if msg_ids is not None:
            messages = [messages[i] for i in msg_ids]
        html_content = '<div style="font-family: Arial, sans-serif;">'
        for message in messages:
            html_content+= message.to_html()
        html_content += '</div>'
        if to_file is not None:
            with open(to_file, "w") as f:
                f.write(html_content)
        return html_content
    def display(self, msg_ids:List[int]=None, theme: str = 'light', to_file: Optional[str] = None):
        html_content = self.to_html(msg_ids, theme, to_file, )
        display(HTML(html_content))

    def get_system_msg(self):
        msgs = self.get_messages()
        if msgs[0].role == 'system':
            system_msg= msgs[0].content
            return system_msg
    def to_sharegpt(self):
        msgs = self.get_messages()
        system_msg = self.get_system_msg()
        if system_msg is None:
            system_msg = 'You are a helpful assistant.'
        else:
            msgs = msgs[1:]
        return {
            'system': system_msg,
            'conversations': [
                {'from': msg.role, 'message': msg.content} for msg in msgs
            ]
        }
        
    def to_xml(self):
        msg = ''
        for turn in self.messages:
            msg += f'<turn_{turn.role}>\n{turn.content}\n</turn_{turn.role}>'
        return msg
        
        
class Conversations(BaseModel):
    conversations: List[Conversation]

    @classmethod
    def from_list(cls, list_items: List[List[dict]]):
        conversations = [Conversation.from_messages_list(list_turns) for list_turns in list_items]
        return cls(conversations=conversations)
    @classmethod
    def from_file(cls, file: str):
        # Ensure `load_chat_dataset` is correctly implemented and available
        from llm_utils import load_chat_dataset
        data = load_chat_dataset(file, current_format='auto', return_format='chatml')
        return cls.from_list(data)

    def to_sharegpts(self):
        return [conversation.to_sharegpt() for conversation in self.conversations]

    def display(self, i):
        self.conversations[i].display()

    def add(self, conversation: Conversation):
        self.conversations.append(deepcopy(conversation))