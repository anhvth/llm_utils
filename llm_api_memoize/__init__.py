from llm_api_memoize import OpenAI
import os
from speedy import memoize_v2


class OpenAIChatMemoize:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    @memoize_v2(['messages', 'model'])
    def call_chat_gpt_api(self, messages, model='gpt-3.5-turbo'):
        openai_response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return openai_response

    def __call__(self, messages, model):
        return self.call_chat_gpt_api(messages, model)
