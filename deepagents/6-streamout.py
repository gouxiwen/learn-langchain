# 多智能体下的流式输出, version="v2"
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool

# from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()

# model = ChatDeepSeek(
#     model="deepseek-chat",
# )

env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model="Qwen/Qwen3-8B",  # 模型名称
    model_provider="openai",  # 模型提供商，硅基流动提供了openai请求格式的访问
    base_url="https://api.siliconflow.cn/v1/",  # 硅基流动模型的请求url
    api_key=os.getenv("siliconflow_api_key"),  # 填写你注册的硅基流动 API Key
    temperature=0.0,
)
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))


@tool
def internet_search(
    query: str,  # 搜索查询字符串，用于指定搜索内容
    max_results: int = 5,  # 返回的最大结果数量，默认为5
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """使用 Tavily API 执行互联网搜索，获取实时或最新的网络信息。

    当需要回答需要当前新闻、最新数据或超出模型知识范围的外部信息时，
    可以使用此工具进行联网搜索。支持普通网页搜索、新闻搜索和金融领域搜索。

    Args:
        query (str): 要搜索的问题或关键词，应清晰、具体地描述所需信息。
        max_results (int, optional): 返回的最大搜索结果数量。默认为 5。
        topic (Literal["general", "news", "finance"], optional): 搜索主题类型。
            - "general"：通用网页搜索，适用于大部分事实性、常识性问题。
            - "news"：新闻搜索，获取近期相关新闻报道。
            - "finance"：金融领域搜索，适用于股票、经济、公司财务等信息。
            默认为 "general"。
        include_raw_content (bool, optional): 是否在结果中包含原始网页正文内容。
            设为 True 会返回更详细的页面文本（可能较长），默认为 False。

    Returns:
        dict: Tavily API 返回的搜索结果对象。通常包含以下字段：
            - "results": 列表，每个元素包含 title、url、content（摘要）等。
            - "query": 原始查询字符串。
            - 若 include_raw_content 为 True，还可能包含 raw_content 字段。
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# 字典方式创建子智能体
research_subagent = {
    "name": "research-agent",
    "description": "用于深度搜索网络信息",
    "system_prompt": "你是一个网络搜索大师，可以调用网络搜索工具搜索用户想了解的内容",
    "tools": [internet_search],
}


# 编译方式创建子智能体
summary_agent = create_agent(
    model=model, system_prompt="你用来根据现有资料总结并提供用户想要的短篇报告"
)

summary_subagent = CompiledSubAgent(
    name="summary-agent",
    description="用来根据提供的新闻或搜索信息编写短篇报告，500字以内",
    runnable=summary_agent,
)

research_instruction = """
 你是一位从事国际关系研究的专家，能够分析不同国家的国情，并按照用户的要求生成报告
"""
agent = create_deep_agent(
    model=model,
    system_prompt=research_instruction,
    subagents=[research_subagent, summary_subagent],
    backend=FilesystemBackend(root_dir="./test_dir", virtual_mode=True),
)


query = "请分析2026年4月21日伊朗和美国战事的情况，并撰写短篇报告分析为什么美国注定失败，500字以内的报告"

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="updates",
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            # 子 Agent 事件，命名空间标明来源
            print(f"[subagent: {chunk['ns']}]")
        else:
            # 主 Agent 事件
            print("[main agent]")
        print(chunk["data"])


# 模式对比与选择指南：

# 模式	         粒度	        输出内容	                                    典型用途
# updates	    节点级别	    每个节点完成后的状态快照	                    追踪执行进度、子 Agent 生命周期
# messages	    Token 级别	    逐 Token 文本 + 工具调用块 + 工具结果	        聊天式 UI、工具调用实时监控
# custom	    自定义	        开发者通过 get_stream_writer() 写入的任意数据	领域特定进度、阶段性通知
# 多模式组合[]	 混合	        以上全部事件类型，按到达顺序交织	              生产级应用、全维度可观测性
