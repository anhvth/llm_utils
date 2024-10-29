import dspy


def inspect_program(program):
    demos = program.demos
    msgs = [{"role": "system", "content": dspy.adapters.chat_adapter.prepare_instructions(program.signature)}]
    for demo in demos:
        for turn in ["user", "assistant"]:
            msg = dspy.adapters.chat_adapter.format_turn(program.signature, demo, turn)
            msgs.append(msg)
    from llm_utils import inspect_msgs
    inspect_msgs(msgs)
