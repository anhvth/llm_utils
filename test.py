text2 = """Example 1:
Task Description: Create a prompt template that takes a query as input and outputs a joke with a setup and punchline.


```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Set up the language model
model = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0.0)

# Define the output structure
class Joke(BaseModel):
    setup: str = Field(description=\"question to set up a joke\")
    punchline: str = Field(description=\"answer to resolve the joke\")

# Create a PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Joke)

# Set up a PromptTemplate with clear instructions
prompt = PromptTemplate(
    template=\\"\"\"
    Create a joke with a setup and punchline.
    
    Setup: A question or statement that sets up the joke.
    Punchline: A humorous answer or twist that resolves the joke.
    
    Please respond in the following format:
    {{
        \"setup\": \"[insert setup here]\",
        \"punchline\": \"[insert punchline here]\"
    }}
    
    Query: {query}
    \\"\"\",
    input_variables=[\"query\"],
)

# Combine the prompt and model
prompt_and_model = prompt | model

# Invoke the program with a sample query
output = prompt_and_model.invoke({\"query\": \"Tell me a joke.\"})
parser.invoke(output)
```

Example 2:
Task Description: Create a prompt template that takes a query as input and outputs a motivational quote.


```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Set up the language model
model = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0.0)

# Define the output structure
class Quote(BaseModel):
    text: str = Field(description=\"motivational quote text\")
    author: str = Field(description=\"author of the quote\")

# Create a PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Quote)

# Set up a PromptTemplate with clear instructions
prompt = PromptTemplate(
    template=\\"\"\"
    Provide a motivational quote.
    
    Please respond in the following format:
    {{
        \"text\": \"[insert quote text here]\",
        \"author\": \"[insert author here]\"
    }}
    
    Query: {query}
    \\"\"\",
    input_variables=[\"query\"],
)

# Combine the prompt and model
prompt_and_model = prompt | model

# Invoke the program with a sample query
output = prompt_and_model.invoke({\"query\": \"Give me a motivational quote.\"})
parser.invoke(output)
```

Example 3:
Task Description: Create a prompt template that takes a query as input and outputs a riddle.


```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Set up the language model
model = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0.0)

# Define the output structure
class Riddle(BaseModel):
    question: str = Field(description=\"the question that forms the riddle\")
    answer: str = Field(description=\"the answer to the riddle\")

# Create a PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Riddle)

# Set up a PromptTemplate with clear instructions
prompt = PromptTemplate(
    template=\\"\"\"
    Create a riddle.
    
    Please respond in the following format:
    {{
        \"question\": \"[insert question here]\",
        \"answer\": \"[insert answer here]\"
    }}
    
    Query: {query}
    \\"\"\",
    input_variables=[\"query\"],
)

# Combine the prompt and model
prompt_and_model = prompt | model

# Invoke the program with a sample query
output = prompt_and_model.invoke({\"query\": \"Give me a riddle.\"})
parser.invoke(output)
```

Example 4:
Task Description: Create a prompt template that takes a query as input and outputs a fun fact.


```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Set up the language model
model = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0.0)

# Define the output structure
class FunFact(BaseModel):
    fact: str = Field(description=\"a fun and interesting fact\")

# Create a PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=FunFact)

# Set up a PromptTemplate with clear instructions
prompt = PromptTemplate(
    template=\\"\"\"
    Provide a fun fact.
    
    Please respond in the following format:
    {{
        \"fact\": \"[insert fact here]\"
    }}
    
    Query: {query}
    \\"\"\",
    input_variables=[\"query\"],
)

# Combine the prompt and model
prompt_and_model = prompt | model

# Invoke the program with a sample query
output = prompt_and_model.invoke({\"query\": \"Tell me a fun fact.\"})
parser.invoke(output)


Example 5: Task Description: Create a prompt template that uses few-shot learning to generate jokes with a setup and punchline, outputting the results in JSON format.


```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# Define the JSON structure we want for our jokes
class Joke(BaseModel):
    setup: str = Field(description=\"question to set up a joke\")
    punchline: str = Field(description=\"answer to resolve the joke\")


# Create some example jokes for few-shot learning
# **NOTE** that example must be List[dict[str, str]]
examples = [
    {
        \"query\": \"Tell me a joke about programming.\",
        \"answer\": \\"\"\"
        {
            \"setup\": \"Why do programmers prefer dark mode?\",
            \"punchline\": \"Because light attracts bugs!\"
        }
        \\"\"\",
    },
    {
        \"query\": \"Give me a joke about animals.\",
        \"answer\": \\"\"\"
        {
            \"setup\": \"Why don\'t oysters donate to charity?\",
            \"punchline\": \"Because they\'re shellfish!\"
        }
        \\"\"\",
    },
]
# *IMPORTANT* format json string
for example in examples:
    for k, v in example.items():
        example[k] = v.replace(\"{\", \"{{\").replace(\"}\", \"}}\")

# Create a template for formatting our examples
example_template = \\"\"\"
Human: {query}
AI: {answer}
\\"\"\"

example_prompt = PromptTemplate(
    input_variables=[\"query\", \"answer\"], template=example_template
)
parser = JsonOutputParser(pydantic_object=Joke)
# Create our few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=\"<system>\\nYou are a helpful AI assistant that generates jokes in JSON format\\n{}\\n</system>. \\n<examples>\".format(
        parser.get_format_instructions().replace(\'{\', \'{{\').replace(\'}\', \'}}\')
    ),
    suffix=\"\\n</examples>\\nHuman:\",
    example_separator=\"\\n\\n\",
).format()

# Set up the output parser


# Create the final prompt template
prompt = PromptTemplate.from_template(
    template=\"{few_shot_examples}{human_query}\\nAI:\",
    partial_variables={\"few_shot_examples\": few_shot_prompt},
)

# # Set up the language model
model = ChatOpenAI(temperature=0, model=\'gpt-4o-mini\')

# # Create the chain
chain = prompt | model | parser

# # Test the chain
result = chain.invoke({\"human_query\": \"Tell me a joke about space.\"})
```"""