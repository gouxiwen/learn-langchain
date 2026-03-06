import requests
from langchain_core.tools import tool
from langchain_classic.chat_models import init_chat_model
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableLambda

@tool
def get_weather(loc):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "StSCw8U4iJyYWfmNb",
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']

# print(get_weather.name)
# print(get_weather.description)
# print(get_weather.args)


# 使用 硅基流动 模型
model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-vcsmjxxaanktozmkjkqkipjsisvbcoezstlvsxslbqqfddir",
)

tools = [get_weather]
llm_with_tools = model.bind_tools(tools)
# response = llm_with_tools.invoke("你好， 请问北京的天气怎么样？")

parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)

# 调试中间结果
def debug_print(x):
    print('中间结果：', x)
    return x

debug_node = RunnableLambda(debug_print)

llm_chain = llm_with_tools |  parser | debug_node | get_weather

response = llm_chain.invoke("你好， 请问北京的天气怎么样？")

print(response)