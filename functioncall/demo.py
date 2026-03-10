# 从GPT-4开始，functioncall是大模型的原生能力，这里演示如何使用大模型调用外部函数
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests

env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
def run_conv(messages,
             api_key,
             tools=None,
             functions_list=None,
             model="glm-4.7"):
    user_messages = messages

    client = OpenAI(api_key=api_key,
                    base_url="https://open.bigmodel.cn/api/paas/v4/")

    # 如果没有外部函数库，则执行普通的对话任务
    if tools == None:
        response = client.chat.completions.create(
            model=model,
            messages=user_messages
        )
        final_response = response.choices[0].message.content

    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # 创建包含用户问题的message
        messages = user_messages

        # first response
        response = client.chat.completions.create(
            model=model,
            messages=user_messages,
            tools=tools,
        )
        response_message = response.choices[0].message

        # 获取函数名
        function_name = response_message.tool_calls[0].function.name
        # 获取函数对象
        fuction_to_call = available_functions[function_name]
        # 获取函数参数
        function_args = json.loads(response_message.tool_calls[0].function.arguments)

        # 将函数参数输入到函数中，获取函数计算结果
        function_response = fuction_to_call(**function_args)

        # messages中拼接first response消息
        user_messages.append(response_message.model_dump())

        # messages中拼接外部函数输出结果
        user_messages.append(
            {
                "role": "tool",
                "content": json.dumps(function_response),
                "tool_call_id": response_message.tool_calls[0].id
            }
        )

        # 第二次调用模型
        second_response = client.chat.completions.create(
            model=model,
            messages=user_messages)

        # 获取最终结果
        final_response = second_response.choices[0].message.content

    return final_response

def get_weather(loc):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.getenv("xinzhi_api_key"),
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']

ds_api_key = os.getenv("zhipu_api_key")
messages = [{"role": "user", "content": "请问上海今天天气如何？"}]
get_weather_function = {
    'name': 'get_weather',
    'description': '查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters': {
        'type': 'object',
        'properties': {  # 参数说明
            'loc': {
                'description': '城市名称',
                'type': 'string'
            }
        },
        'required': ['loc']  # 必备参数
    }
}
tools = [
    {
        "type": "function",
        "function": get_weather_function
    }
]
final_response = run_conv(messages=messages,
         api_key=ds_api_key,
         tools=tools,
         functions_list=[get_weather])
print(final_response)