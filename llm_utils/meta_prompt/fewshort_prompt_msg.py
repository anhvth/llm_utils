from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
import json
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.pydantic_v1 import BaseModel as BaseModelV1
from typing import List, Dict, Tuple, Sequence
from llm_utils import get_langchain_openai_model
from langchain.output_parsers import PydanticOutputParser


def get_few_shot_prompt(
    examples_human_ai_pairs,
    system_message: str = "",
    format_inst="",
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
        if isinstance(example, tuple):
            role, msg = example
            role = HumanMessage if role == "human" else AIMessage
            msgs.append(role(msg))

        else:
            human_message = example["human"]
            ai_message = example["ai"]
            msgs.append(HumanMessage(human_message))
            msgs.append(AIMessage(json.dumps(ai_message, indent=2)))

    msgs.extend(few_shot_messages)
    if format_inst:
        msgs.append(("system", format_inst))
    msgs.append(("human", "{input}"))
    final_prompt = ChatPromptTemplate.from_messages(msgs)
    return final_prompt


def generate_pydantic_parse_chain(
    examples, pydantic_output_model=None, model="gpt-4o-mini", system_message=""
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
    prompt = get_few_shot_prompt(examples, system_message, format_inst)
    model = get_langchain_openai_model(model)
    return prompt, model, parser
