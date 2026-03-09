# langgraph api按照封装程度分三层
# 这里展示顶层预构建api和中间层node api调用示例，展示了如何使用langgraph预制图结构+自定义工具创建一个智能体
# 多工具调用示例
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# 使用pydantic库定义一个对象类型描述传入参数，这里表示要传入的是一个字符串loc参数，表示的含义是城市名称
class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")

class WriteQuery(BaseModel):
    content: str = Field(description="需要写入文档的具体内容")

# 定义的WeatherQuery对象在@tool(args_schema=WeatherQuery)中约束get_weather的函数参数
@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "自己的apikye",
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']

@tool(args_schema=WriteQuery)
def write_file(content):
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return：是否成功写入
    """
    with open('res.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    return "已成功写入本地文件。"

model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key='自己的apikey',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 多工具
tools = [get_weather, write_file]

# 创建ReACT预制图结构并构建智能体
# ReACT预制图结构：re，reason，思考，act，action，行动，它的工作流程是一个循环图：思考->行动->观察结果->再思考...->得出答案
agent = create_react_agent(model=model, tools=tools)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "北京和杭州现在的天气如何?并把查询结果写入文件中"
            }
        ]
    },
    {
        "recursion_limit": 4
    },
)

# print(response)
print(response['messages'][-1].content)