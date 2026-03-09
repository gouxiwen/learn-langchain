# langgraph api按照封装程度分三层
# 这里展示底层 api调用示例
# 添加大模型节点

from typing import Annotated,TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests


# 使用TypedDict创建具有特定键值类型的字典类来规范状态结构，其中messages键是一个列表，我们通过Annotated注解指定使用add_messages函数来处理消息的累积更新。
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(AgentState)


# 定义工具
class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


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
model = model.bind_tools(tools)

# 调用大模型节点
def call_model(
    state: AgentState,
):
    system_prompt = SystemMessage(
        "你是一个AI助手，可以依据用户提问产生回答，你还具备调用天气函数的能力"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph = graph.compile()

final_state = graph.invoke({"messages": ["请问上海天气如何?"]})
print(final_state['messages'])