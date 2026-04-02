# langgraph的设计模式：
# 提示链模式、路由模式、并行模式比较简单不写代码，这里展示协调器-工作者模式。
# 协调器-工作者模式代码实战，由协调器通过Send API动态分配工作者完成任务，适用于需要将复杂任务分解为多个子任务的场景。
import operator
import os
from pathlib import Path
from typing import Annotated, List, TypedDict

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
# 1. 定义结构化的输出格式，用于规划报告的章节
class Section(BaseModel):
    name: str = Field(description="报告章节的名称")
    description: str = Field(description="本章节中涵盖的主要主题和概念的简要概述")


class Sections(BaseModel):
    sections: List[Section] = Field(description="报告的章节")


planner = llm.with_structured_output(Sections)


# 2. 定义协调器的状态
class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str


# 3. 定义工作者状态
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# 4. 定义图中的节点:
def orchestrator(state: State):
    """
    协调器节点，使用结构化输出生成报告计划
    """
    topic = state["topic"]
    # 首先使用planner生成结构化输出报告
    report_sections = planner.invoke(
        [
            SystemMessage(content="请根据用户输入的主题生成报告计划。"),
            HumanMessage(content=f"这是报告主题:{topic}"),
        ]
    )
    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """
    工作者节点：根据分配的章节详细信息描述生成报告的章节内容
    """
    section_name = state["section"].name
    section = llm.invoke(
        [
            SystemMessage(
                content="根据提供的章节的名称和描述编写报告章节，每个章节中不包含序言，使用markdown格式。200字以内"
            ),
            HumanMessage(content=f"这是章节的名称: {section_name}"),
        ]
    )
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """
    将各个章节的输出合称为完整的报告
    """
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}


# 5. 定义条件边函数，将工作者【动态】分配对应的计划章节
def assign_workers(state: State):
    """
    使用send API 将工作者分配给计划中的每个章节，以实现动态工作者创建
    """
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()


result = orchestrator_worker.invoke({"topic": "创建关于LLM缩放定律的报告"})

print(result["final_report"])
