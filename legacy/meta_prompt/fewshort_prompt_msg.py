from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
import json
from langchain_core.prompts.chat import MessageLikeRepresentation
from pydantic import BaseModel as BaseModelV1
from typing import List, Dict, Tuple, Sequence

from langchain.output_parsers import PydanticOutputParser


class Example(BaseModelV1):
    human: str
    ai: str


def get_few_shot_prompt(
    examples_human_ai_pairs: List[Example],
    system_message: str = "",
    format_inst="",
    ai_prefix="",
) -> ChatPromptTemplate:

    # assert len(examples_human_ai_pairs) > 0
    """
    Generate a chat prompt template for few-shot learning.
    Args:
        examples_human_ai_pairs (List[Dict[str, str]]): A list of dictionaries representing human-AI pairs.
            Each dictionary should have keys 'human' and 'ai' representing the human and AI messages, respectively.
        system_message (str, optional): A system message to be included in the prompt. Defaults to "".
    Returns:
        ChatPromptTemplate: The generated chat prompt template.
    Raises:
        AssertionError: If the length of examples_human_ai_pairs is less than or equal to 0.
        AssertionError: If the keys of the first dictionary in examples_human_ai_pairs are not {'human', 'ai'}.
    """
    # assert examples_human_ai_pairs[0].keys() == {"human", "ai"}
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = []
    if system_message:
        msgs.append(("system", system_message))
    few_shot_messages = []
    for example in examples_human_ai_pairs:
        msgs.append(HumanMessage(example.human))
        msgs.append(AIMessage(example.ai))

    msgs.extend(few_shot_messages)
    if format_inst:
        msgs.append(("system", format_inst))
    msgs.append(("human", "{input}"))
    if ai_prefix:
        msgs.append(("ai", ai_prefix))
    final_prompt = ChatPromptTemplate.from_messages(msgs)
    return final_prompt


def generate_pydantic_parse_chain(
    examples: List[Example],
    pydantic_output_model=None,
    model="gpt-4o-mini",
    system_message="",
    ai_prefix="",
):
    if pydantic_output_model is None:
        format_inst = ""
        parser = None
    else:
        parser = PydanticOutputParser(pydantic_object=pydantic_output_model)
        format_inst = json.dumps(parser.get_format_instructions(), indent=2)
        format_inst = (
            parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        )
    prompt = get_few_shot_prompt(
        examples, system_message, format_inst, ai_prefix=ai_prefix
    )
    from ._generate_prompt import get_langchain_openai_model

    model = get_langchain_openai_model(model)
    return prompt, model, parser
