# langgraph api按照封装程度分三层
# 这里展示顶层预构建api和中间层node api调用示例，展示了如何使用langgraph预制图结构+自定义工具创建一个智能体
# 单工具调用示例
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# 使用pydantic库定义一个对象类型描述传入参数，这里表示要传入的是一个字符串loc参数，表示的含义是城市名称
class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")

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
        "key": "StSCw8U4iJyYWfmNb",
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']


model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key='75301e9d6ffc4d878a32a2a5b31dc8c0.frRvWZTAQklAYIXJ',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

tools = [get_weather]

# 创建ReACT预制图结构并构建智能体
# ReACT预制图结构：re，reason，思考，act，action，行动，它的工作流程是一个循环图：思考->行动->观察结果->再思考...->得出答案
# 值得注意的是：ReACT使用方法和langchain中create_tool_calling_agent+AgentExecutor类似
agent = create_react_agent(model=model, tools=tools)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "北京现在的天气如何?"
            }
        ]
    }
)

# print(response)
print(response['messages'][-1].content)