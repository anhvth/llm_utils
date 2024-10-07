from copy import deepcopy
from typing import List, Literal, Optional, Union

from IPython.display import HTML, display
from loguru import logger
from pydantic import BaseModel


class Message(BaseModel):

    def __init__(self, **data):
        # Check if the role is 'human' and change it to 'user'
        if data.get("role") == "human":
            data["role"] = "user"
        super().__init__(**data)

    role: Literal["user", "assistant", "system", "function"]
    content: str

    def to_html(self):
        message = self
        color_scheme = {
            "system": {"background": "#FFAAAA", "text": "#000000"},
            "user": {"background": "#AAFFAA", "text": "#000000"},
            "assistant": {"background": "#AAAAFF", "text": "#000000"},
        }

        # Get colors for the current message
        background_color, text_color = color_scheme.get(
            message.role, {"background": "#FFFFFF", "text": "#000000"}
        ).values()

        # Safe message content
        content_safe = (
            message.content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>").replace(" ", "&nbsp;")
        )

        # HTML for the message
        html_content = ""
        html_content += (
            f'<div style="background-color:{background_color}; margin:10px; padding:10px; border-radius:8px;">'
        )
        html_content += f'<strong style="color:{text_color};">{message.role.capitalize()}:</strong> <span style="color:{text_color};">{content_safe}</span>'
        html_content += "</div>"
        return html_content


from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_community.adapters.openai import convert_openai_messages
from pydantic import BaseModel
from typing import List, Optional


class Conversation(BaseModel):
    messages: List[Message] = []

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def get_messages(self) -> List[Message]:
        return self.messages

    @classmethod
    def from_messages(cls, messages_list: List[Union[HumanMessage, SystemMessage, AIMessage, dict, tuple]]):
        messages = []
        for msg_item in messages_list:
            try:
                # Handle LangChain message types
                if isinstance(msg_item, (HumanMessage, SystemMessage, AIMessage)):
                    role = "user" if isinstance(msg_item, HumanMessage) else "assistant"
                    messages.append(Message(role=role, content=msg_item.content))

                # Handle dictionaries
                elif isinstance(msg_item, dict):
                    messages.append(Message(**msg_item))

                # Handle tuples
                elif isinstance(msg_item, tuple) and len(msg_item) == 2:
                    role, content = msg_item
                    messages.append(Message(role=role, content=content))

                else:
                    raise ValueError(f"Invalid message format: {msg_item}")

            except Exception as e:
                logger.error(f"Error while parsing message {msg_item}: {e}, this message will be ignored")

        return cls(messages=messages)

    def to_langchain(self) -> List[BaseMessage]:
        langchain_messages = []
        for message in self.messages:
            if message.role == "user":
                langchain_messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                langchain_messages.append(AIMessage(content=message.content))
            elif message.role == "system":
                langchain_messages.append(SystemMessage(content=message.content))
            else:
                raise ValueError(f"Unknown role '{message.role}' in message: {message}")
        return langchain_messages

    def to_dict(self) -> dict:
        return {"messages": [message.model_dump() for message in self.messages]}

    def to_list(self):
        return self.to_dict()["messages"]

    def to_html(self, msg_ids: List[int] = None, theme: str = "light", to_file: Optional[str] = None):
        messages = self.get_messages()
        if msg_ids is not None:
            messages = [messages[i] for i in msg_ids]
        html_content = '<div style="font-family: Arial, sans-serif;">'
        for message in messages:
            html_content += message.to_html()
        html_content += "</div>"

        if to_file is not None:
            with open(to_file, "w") as f:
                f.write(html_content)

        return html_content

    def display(self, msg_ids: List[int] = None, theme: str = "light", to_file: Optional[str] = None):
        html_content = self.to_html(msg_ids, theme, to_file)
        display(HTML(html_content))

    def get_system_msg(self):
        msgs = self.get_messages()
        if msgs and msgs[0].role == "system":
            return msgs[0].content
        return None

    def to_sharegpt(self):
        msgs = self.get_messages()
        system_msg = self.get_system_msg() or "You are a helpful assistant."

        if self.get_system_msg() is not None:
            msgs = msgs[1:]  # Remove the first message if it's a system message

        return {"system": system_msg, "conversations": [{"from": msg.role, "message": msg.content} for msg in msgs]}

    def to_xml(self):
        msg = ""
        for turn in self.messages:
            msg += f"<turn_{turn.role}>\n{turn.content}\n</turn_{turn.role}>\n"
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

        data = load_chat_dataset(file, current_format="auto", return_format="chatml")
        return cls.from_list(data)

    def to_sharegpts(self):
        return [conversation.to_sharegpt() for conversation in self.conversations]

    def display(self, i):
        self.conversations[i].display()

    def add(self, conversation: Conversation):
        self.conversations.append(deepcopy(conversation))
