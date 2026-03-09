# langgraph api按照封装程度分三层
# 这里展示底层 api调用示例
# 添加大模型节点

from typing import Annotated,TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.constants import START
from langchain_core.messages import AIMessage, HumanMessage


# 使用TypedDict创建具有特定键值类型的字典类来规范状态结构，其中messages键是一个列表，我们通过Annotated注解指定使用add_messages函数来处理消息的累积更新。
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key='75301e9d6ffc4d878a32a2a5b31dc8c0.frRvWZTAQklAYIXJ',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 定义大模型节点
def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

# 添加节点
graph_builder.add_node("chatbot", chatbot)

# 添加边
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
# 单论对话
# final_state = graph.invoke({"messages": ["你好，我叫陈明，好久不见。"]})
# print(final_state['messages'])

# 多轮对话
messages_list = [
    HumanMessage(content="你好，我叫大模型真好玩，好久不见。"),
    AIMessage(content="你好呀！我是苍老师，是一名女演员。很高兴认识你！"),
    HumanMessage(content="请问，你还记得我叫什么名字么？"),
]
final_state = graph.invoke({"messages": messages_list})
print(final_state['messages'])

