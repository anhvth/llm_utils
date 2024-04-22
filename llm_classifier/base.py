from pydantic import BaseModel, Field
from llm_utils import get_conversation_one_turn, display_chat_messages_as_html
import json
from speedy import memoize, identify, dump_json_or_pickle, load_by_ext
from typing import *
import os
from loguru import logger
from glob import glob
from pydantic import BaseModel, create_model
import json
from typing import Type
from llm_utils import extract_json
CACHE_DIR = os.path.expanduser("~/.cache/llm_classifier")
logger.info(f'LLM_CLASSIFIER_CACHE_DIR: {CACHE_DIR}')

@memoize
def embeding_fn(text, text_embeding_model="text-embedding-3-small") -> List[float]:
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        input=text,
        model=text_embeding_model,
    )

    return response.data[0].embedding


# Define a Pydantic model
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tags: List[str] = []


# Function to load a Pydantic model schema from a file and create a model class
def load_schema(path: str) -> Type[BaseModel]:
    with open(path, 'r') as f:
        schema = json.load(f)
    # Dynamically create a new model class from the loaded schema
    model_name = schema.get('title', 'DynamicModel')
    fields = {name: (field_info.get('type'), field_info.get('default'))
                for name, field_info in schema.get('properties', {}).items()}
    return create_model(model_name, **fields)



LABEL_BY_PRIORITY = ['human', 'gpt4', 'gpt3']

class Example(BaseModel):
    input: str
    output: Union[str, dict, BaseModel]
    hint: Optional[str] = None
    label_by: str = None

    def model_dump_json(self):
        return {
            'input': self.input,
            'output': self.output if not isinstance(self.output, BaseModel) else json.loads(self.output.model_dump_json()),
            'hint': self.hint
        }

    def __str__(self):
        output = self.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except:
                raise ValueError(f'expected_output is not a valid json: {output}')
        elif isinstance(output, BaseModel):
            output = output.model_dump_json()
        else:
            assert isinstance(output, dict), f'expected_output must be a dict, got {type(output)}'
        output = '```json\n' + json.dumps(output, indent=2, ensure_ascii=False) + '\n```'
        out_str = f'User: {self.input}\nAssistant: {output}'
        if self.hint:
            out_str += f'\nHint: {self.hint}'
        return out_str

    def __repr__(self):
        output_dict = self.output.model_dump_json() if isinstance(self.output, BaseModel) else self.output
        return '''Example(input="{}", output={}, hint="{}")'''.format(self.input, output_dict, self.hint)


def merge_examples(examples, new_examples):
    examples_map = {example.input: example for example in examples}
    new_examples_map = {example.input: example for example in new_examples}
    inputs = set(examples_map.keys()) | set(new_examples_map.keys())
    new_examples = []
    for input in inputs:
        example = examples_map.get(input)
        new_example = new_examples_map.get(input)
        if example is None:
            new_examples.append(new_example)
        elif new_example is None:
            new_examples.append(example)
        else:
            if LABEL_BY_PRIORITY.index(new_example.label_by) < LABEL_BY_PRIORITY.index(example.label_by):
                new_examples.append(new_example)
            else:
                new_examples.append(example)
    new_examples.sort(key=lambda x: LABEL_BY_PRIORITY.index(x.label_by))
    return new_examples

class LLMWraper:
    def __init__(self, model, name) -> None:
        self.model = model
        self.name = name
    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

class BaseExtractor:
    def __init__(self, msgs_forward_fn:callable, task_description: str,
                examples: list, pydantic_response: BaseModel, 
                classifier_path, note: str = None, max_examples=10):
        self.classifier_path = classifier_path
        
        self.max_examples = max_examples
        self.msgs_forward_fn = msgs_forward_fn
        self.task_description = task_description
        self.examples = examples
        self.pydantic_response = pydantic_response
        self.note = note
        self.system_msg = self._create_system_msg()
        self.update()

    def _create_system_msg(self, examples=None):
        examples = examples or self.examples
        # assert len(examples) < 10, 'The number of examples should be less than 10'
        if len(examples) > self.max_examples:
            examples = examples[:self.max_examples]
        system_msg = f'''
<role>
{self.task_description}
</role>

<examples>
Below are some examples of the conversation between the user and the assistant. The user provides an input prompt, and the assistant responds with a JSON object.
'''
        for i, example in enumerate(examples, start=1):
            system_msg += f'''
<example id={i}>
{str(example)}
</example>
'''
        system_msg += '''
</examples>
'''

        if self.note:
            system_msg += f'''
<note>
{self.note}
</note>
'''

        if self.pydantic_response:
            # Retrieve the name of the Pydantic model class
            model_name = self.pydantic_response.__class__.__name__
            # Generate the JSON schema for the Pydantic model
            model_definition = self.pydantic_response.model_json_schema()
            # Append a formatted message with instructions and the Pydantic model schema

        system_msg += f'''
<pydantic>
To ensure compatibility, please format your response as JSON according to the schema of the Pydantic model '{model_name}'. Below is the JSON schema that your response should adhere to:
```json
{model_definition}
```
</pydantic>

'''

        return system_msg

    def __call__(self, input_data: str, hint:str=None):
        user_msg = f"'''{input_data}'''"
        if hint:
            user_msg += f'\nHint: {hint}'
        msgs = get_conversation_one_turn(self.system_msg, user_msg)
        self.last_msgs = msgs = self.msgs_forward_fn(msgs)
        json_out = extract_json(msgs[-1]['content'])

        if self.pydantic_response:
            try:
                return self.pydantic_response(**json_out)
            except Exception as e:
                from loguru import logger
                logger.warning(f'Error validating response: {e}, response: {json_out}')
        return json_out

    def display(self):
        display_chat_messages_as_html(self.last_msgs)

    # def add_example(self, example: Example):
    #     self.examples.append(example)
    #     self.system_msg = self._create_system_msg()

    def get_predict_example(self):
        d = extract_json(self.last_msgs[-1]['content'])
        input = self.last_msgs[-2]['content']
        input = input[3:-3]
        return Example(input=input, output=self.pydantic_response(**d))

    def detect_wrong_example(self, feedback, msgs_forward_fn=None):
        if msgs_forward_fn is None:
            msgs_forward_fn = self.msgs_forward_fn
        system_prompt = '''You are now tasked improved the current system prompt.
You will be provided the current_system_prompt, the user input and the model prediction along with the user feedback on that example.

Response in json format with the following pydanctic model:
class Action(BaseModel):
    function: str
    args: dict
class Response(BaseModel):
    actions: List[Action]

- Step 1: Brainstorm ways to improve the system prompt based on the user feedback (less than 200 words)
- Step2 2: Response the list of action in JSON format with the list function calls to update the system prompt.

Tools:
You have access to the following functions to update the current system prompt:
    - remove_example(id: int): Remove the example with the specified id.
    - add_example(example: Example): Add a new example to the list of examples.
    - update_instruction(new_instruction: str): Update the instruction for the task.
    - modify_example(id: int, new_example: Example): Modify the example with the specified id.
'''
        user_msg = f'''
<current_system_prompt>
{self.system_msg}
</current_system_prompt>
<wrong_example>
{self.get_predict_example()}
</wrong_example>
<feedback>
{feedback}
</feedback>
        '''
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_msg},
        ]
        return msgs_forward_fn(msgs)

    def get_similar_examples(self, input:str, topk=5):
        from scipy.spatial.distance import cosine
        from speedy import multi_thread
        import pandas as pd
        example_input = [example.input for example in self.examples]
        embedings = multi_thread(embeding_fn, example_input, 32)
        df = pd.DataFrame({"input": example_input, "embeddings": embedings})
        q_embeddings = embeding_fn(input)
        df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))
        # get the topk similar examples
        ids = df.sort_values("distances").head(topk).index
        return [self.examples[id] for id in ids]

    def update_system_prompt_by_question(self, questions:List[str]|str, k=10):
        if isinstance(questions, str):
            questions = [questions]
        examples = []
        example_per_question = k // len(questions)
        example_per_question = 1 if example_per_question == 0 else example_per_question
        example_per_question = min(example_per_question, 3)
        for question in questions:
            examples.extend(self.get_similar_examples(question, example_per_question))
        self.system_msg = self._create_system_msg(examples)
        
    @property
    def database_path(self):
        # save this whole object to a file, so that we can load it later
        class_name = self.__name__
        example_id = identify(self.examples)
        # version is the number of files with the same name

        output_dir = os.path.join(CACHE_DIR, f'{class_name}/')

        version_id = len(glob(os.path.join(output_dir, 'version_*.json')))+1
        output_path = os.path.join(output_dir, f'version_{version_id}_{example_id}.json')
        return output_path

    @classmethod
    def load(cls, path=None):
        # load task_description, examples, pydantic_response, note
        data = json.load(open(path))
        examples = [Example(**example) for example in data['examples']]
        if data['pydantic_response']:
            pydantic_response = BaseModel.model_validate(data['pydantic_response'])
        else:
            pydantic_response = None

        note = data['note']
        return cls(task_description=data['task_description'], examples=examples, pydantic_response=pydantic_response, note=note)

    def update(self):
        on_memory_examples = self.examples
        if os.path.exists(self.classifier_path):
            data = load_by_ext(self.classifier_path)
            on_disk_examples = data['examples']
            print(data.keys())
            task_description = data['task_description']
            if task_description != self.task_description:
                logger.warning(f'Task description does not match with the on-disk task description: {self.task_description} != {task_description}')
                self.task_description = task_description
            on_disk_examples = [Example(**example) for example in on_disk_examples]
            examples = merge_examples(on_memory_examples, on_disk_examples)
            self.examples = examples
            self.system_msg = self._create_system_msg()
            self.stats()

    def persis(self):
        self.examples = [example.model_dump_json() for example in self.examples]

    def stats(self):
        # print number of examples and by the label_by
        from collections import Counter
        label_by = [example.label_by for example in self.examples]
        label_by_counter = Counter(label_by)
        print(f'Number of examples: {len(self.examples)}')
        print(f'Label by: {label_by_counter}')


class ListExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_system_msg(self, examples=None):
        system_msg = super()._create_system_msg(examples)
        system_msg += '''
<list_input>
The input provided by the user will be an enumerated list. Please process each item in the list separately and provide the response as a list of JSON objects, where each object adheres to the specified Pydantic model schema.
</list_input>
'''
        return system_msg

    def __call__(self, input_data: list[str], hint=None, return_examples=True):
        if isinstance(input_data, str):
            input_data = [input_data]
        user_msg = "Please process the following list of inputs:\n"
        for i, item in enumerate(input_data, start=1):
            user_msg += f"input_id: {i}. {item}\n"
        if hint:
            user_msg += f'\nHint: {hint}'
        msgs = get_conversation_one_turn(self.system_msg, user_msg)
        self.last_msgs = msgs = self.msgs_forward_fn(msgs)
        outputs = msgs[-1]['content']
        if self.pydantic_response:
            try:
                is_list = len(input_data) > 1
                outputs = extract_json(outputs, is_list=is_list)
                if not is_list:
                    outputs = [outputs]
                if return_examples:
                    outputs = self.get_examples(input_data, outputs)
            except Exception as e:
                from loguru import logger
                logger.warning(f'Error validating response: {e}, response: {outputs}')
                raise

        return outputs

    def get_examples(self, inputs:List[str], json_out_list:List[dict]):
        examples = []
        for input, json_out in zip(inputs, json_out_list):
            examples.append(Example(input=input, output=self.pydantic_response(**json_out)))
        return examples