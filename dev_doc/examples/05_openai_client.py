import json

import requests


def demo_tools():
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get current temperature at a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_temperature_date",
                "description": "Get temperature at a location and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "date": {
                            "type": "string",
                            "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location", "date"],
                },
            },
        },
    ]
    MESSAGES = [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
        {"role": "user", "content": "What's the temperature in BeiJing now? How about tomorrow?"},
        # {"role": "user", "content": "What's the temperature in BeiJing now? "},
    ]
    return MESSAGES, TOOLS


def demo_payload(messages: list, tools: list | None):
    payload = {
        "model": "/data/models/Qwen2.5-0.5B-Instruct",
        "messages": messages,
        "tools": tools,
        "top_p": 0.95,
        "top_k": 40,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    return payload


def chat_qwen3(messages: list, tools: list | None):
    url = "http://101.126.140.232:9071/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = demo_payload(messages, tools)
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # 打印返回结果
    print(response.status_code)
    # print(response.text)
    return response.text


def _check_llm_ok():
    messages = [{"role": "user", "content": "写一个笑话。"}]
    txt = chat_qwen3(messages, None)
    print(txt)


def _test_fc():
    messages, functions = demo_tools()
    txt = chat_qwen3(messages, functions)
    print(txt)


if __name__ == "__main__":
    # _check_llm_ok()
    _test_fc()
