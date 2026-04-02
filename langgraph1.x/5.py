# 评估器-优化器模式
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
from langchain.chat_models import init_chat_model

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
    model="glm-4.5",
    api_key=os.getenv("zhipu_api_key"),
)


# 定义结构化的输出：主要用来定义报告的章节。
# 此处使用Pydantic模型进行约束，通过Field为不同字段提供描述，可以更有效地帮助大模型规划器
# 1. 定义评估器的结构
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="判断笑话是否有趣")
    feedback: str = Field(description="如果笑话不好笑，提供改进它的反馈")


evaluator = llm.with_structured_output(Feedback)


# 2. 定义状态
class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    funny_or_not: str


# 3. 定义节点
def llm_call_generator(state: State):
    """
    生成器节点，llm生成笑话，可能会结合之前评估器的反馈
    """
    topic = state["topic"]
    if state.get("feedback"):
        feedback = state["feedback"]
        msg = llm.invoke(f"请写一个关于{topic}的笑话，但是要考虑反馈:{feedback}")
    else:
        msg = llm.invoke(f"写一个关于{topic}的笑话")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """
    评估生成笑话
    """
    joke = state["joke"]
    grade = evaluator.invoke(f"评估笑话{joke}是否好笑,如果不好笑给出修改建议")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# 定义条件边
def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected"


optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {"Accepted": END, "Rejected": "llm_call_generator"},
)
optimizer_workflow = optimizer_builder.compile()

# 运行测试
result = optimizer_workflow.invoke({"topic": "贾乃亮与pg one"})
print(result["joke"])
