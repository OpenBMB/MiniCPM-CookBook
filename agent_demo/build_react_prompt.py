import json
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following tools:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""

def build_input_text(chat_history, list_of_plugin_info) -> str:
    # 候选插件的详细信息
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get("args_format", "json") == "json":
            tool += " Format the arguments as a JSON object."
        elif plugin_info["args_format"] == "code":
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = "\n\n".join(tools_text)

    # 候选插件的代号
    tools_name_text = ", ".join(
        [plugin_info["name_for_model"] for plugin_info in list_of_plugin_info]
    )

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip(
            "\n"
        ).rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        response = response.lstrip(
            "\n"
        ).rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}"
    return prompt

def parse_latest_plugin_call(text):
    plugin_name, plugin_args = "", ""
    if text.startswith("\nAction:"):
        i = text.rfind("\nAction:")
        action_string = "\nAction:"
    else:
        i = text.find("Action:")
        action_string = "Action:"
    if "\nAction Input:" in text:
        j = text.rfind("\nAction Input:")
        action_input_string = "\nAction Input:"
    else:
        j = text.rfind("Action Input:")
        action_input_string = "Action Input:"
    if "\nObservation:" in text:
        k = text.rfind("\nObservation:")
        observation_string = "\nObservation:"
    else:
        k = text.rfind("Observation:")
        observation_string = "Observation:"
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + observation_string  # Add it back.
        k = text.rfind(observation_string)
        plugin_name = text[i + len(action_string) : j].strip()
        plugin_args = text[j + len(action_input_string) : k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text
