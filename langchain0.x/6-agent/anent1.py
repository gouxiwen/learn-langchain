# LangChain对于Agent的定义是“由大模型规划并自由组装各种链来满足用户需求”
import requests

from langchain_classic.agents import create_tool_calling_agent, tool, AgentExecutor
from langchain_classic.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

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

# 构建提示模版， 提示词模板对于Agent的构建是必须的
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息,并具备将结果写入文件的能力"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # 这部分agnet提示符写法是写死的不可以修改
    ]
)

# 使用 智普 模型
model = init_chat_model(
    # model="glm-5",
    model="glm-4.7",
    model_provider="openai", # 模型提供商，智普提供了openai请求格式的访问
    openai_api_key="75301e9d6ffc4d878a32a2a5b31dc8c0.frRvWZTAQklAYIXJ",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
# response=model.invoke("你好")


#定义工具
tools = [get_weather]

# 直接使用`create_tool_calling_agent`创建代理
agent = create_tool_calling_agent(model, tools, prompt)

# 使用AgentExecutor运行当前Agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Verbose表示是否要打印执行细节
# response=agent_executor.invoke({"input":"请问今天北京天气怎么样？"})

# 并联调用
response=agent_executor.invoke({"input":"请问今天北京和杭州的天气怎么样，哪个城市更热？？"})

print(response)