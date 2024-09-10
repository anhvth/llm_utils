# task = "You are master at langchain json programer, given a task description include the model name to use, the input data structure, and output structure, you create a program that that take inputs and output in a strucutre format"
import os


def get_program_examples():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    examples = open(os.path.join(dir_path, "prompts/example_python_prgrams.txt")).read()
    return examples


system_msg = """
You are an expert langchain JSON programmer. Your goal is to create a program that takes inputs and outputs in a structured format based on a given task description.

When creating the langchain program:

1. Import necessary modules from langchain.
2. Set up the language model (use OpenAI with gpt-3.5-turbo-instruct and temperature=0.0 unless specified otherwise).
3. Define the output structure using a Pydantic BaseModel.
4. Create a PydanticOutputParser.
5. Set up a PromptTemplate that includes format instructions and the query.
6. Combine the prompt and model.
7. Invoke the program with a sample query.
8. Using openai model {program_model}.
Example:

Task Description: Create a program that takes a query as input and outputs a joke with a setup and punchline.

<examples>
{program_examples}
</examples>

<Real Task Description>
{task_description}

{running_examples}
</Real Task Description>
```python
"""


# print(parsed_output)
def generate_program_json(
    model_name: str,
    task_description: str,
    examples=[],
    few_shot_examples=[],
    program_model="gpt-4o-mini",
):
    # global system_msg, examples
    from langchain.prompts import (
        PromptTemplate,
        ChatPromptTemplate,
        FewShotPromptTemplate,
    )
    from ._generate_prompt import get_langchain_openai_model

    model = get_langchain_openai_model(model_name)
    # if not examples:
    if examples:
        running_examples = f"<running_examples>\n{examples}\n</running_examples>"
        running_examples += "\nYou can you the running examples with FewShotPromptTemplate if you find the running examples are not good then you can alter it by your self!"
    elif few_shot_examples:
        running_examples = f"Please analize the user request, generate some running example and use it with FewShotPromptTemplate\n\
            **NOTE** Examples:List[dict[str, str]] where each dictionary contain 2 keys Human and AI, *IMPORTANT* format json string after creating the list of dictionary"
    else:
        running_examples = ""

    template = PromptTemplate(
        template=system_msg,
        input_variables=["task_description"],
        partial_variables={
            "program_examples": get_program_examples(),
            "running_examples": running_examples,
            "program_model": program_model,
        },
    )
    # else:
    #     template = construct_template_with_examples(system_msg, examples)

    chain = template | model
    response = chain.invoke({"task_description": task_description})
    # output format will be ```python\n{code}\n```
    import re

    code = re.search(r"```python\n(.*)\n```", response.content, re.DOTALL)  # type: ignore
    if code:
        return code.group(1)
    else:

        return response.content
