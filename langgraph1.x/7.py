# 对于多智能体场景，就需要使用架构来解决问题
# 这里演示网络架构。
# 为了简化开发，LangChain 团队推出了 langgraph-swarm 类库，它封装了用于创建具备移交能力的智能体工具，使得多个智能体能够以对等的方式进行交互与协作。
import operator
import os
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict

from langchain.messages import SystemMessage, HumanMessage

from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver

env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
# llm = init_chat_model(
#     model="glm-4.5",
#     model_provider="openai",
#     api_key=os.getenv("zhipu_api_key"),
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
# )
# init_chat_model初始化的智普模型不支持结构化输出，所以这里直接使用ChatZhipuAI来调用智普模型的接口实现结构化输出的功能。
# glm4.7并发限制是2，glm4.5并发限制是10
llm = ChatZhipuAI(
    model="glm-5",
    api_key=os.getenv("zhipu_api_key"),
)


@tool
def add(a: int, b: int) -> int:
    """
    计算两个整数和字符串相加时务必调用该函数
    """
    print("Agent1 加法工具调用")
    return a + b


# agent1是一位可以调用add函数和移交给agent2移交工具函数的智能体
agent1 = create_agent(
    model=llm,
    tools=[
        add,
        create_handoff_tool(
            agent_name="agent2", description="当用户想和agent2对话时，转给agent2回答"
        ),
    ],
    system_prompt="你是agent1，一位加法专家，可以利用提供的add函数完成所有加法运算",
    name="agent1",
)

# agent2说话的语气像小猫咪，并且拥有一个移交给agent1以寻求数学帮助的移交工具
agent2 = create_agent(
    model=llm,
    tools=[
        create_handoff_tool(
            agent_name="agent1",
            description="请务必将所有的加法运算移交给agent1, 它可以帮助你解决数学问题",
        )
    ],
    system_prompt="你是agent2， 你说话语气像小猫咪",
    name="agent2",
)


# Swarm架构需要通过记忆保持对话连续性，这里使用内存检查点InMemorySaver来实现短期记忆
checkpointer = InMemorySaver()
workflow = create_swarm(
    [agent1, agent2], default_active_agent="agent1"  # 默认激活的智能体是agent1
)

app = workflow.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}

# 第一轮对话
first = app.invoke(
    {"messages": [{"role": "user", "content": "我想和agent2说话，请转接agent2"}]},
    config,
)

print(first["messages"][-1].content)  # 第一轮输出
print("\n\n")

# 第二轮对话
second = app.invoke(
    {"messages": [{"role": "user", "content": "100+100等于多少"}]}, config
)
print(second["messages"][-1].content)  # 第二轮输出
