# 两种方式创建自定义子智能体
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
    model="glm-4.5-air",
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.0,
)
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))


@tool
def internet_search(
    query: str,
    max_results: int = 5,
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


step_num = 0
query = "请分析2026年4月21日伊朗和美国战事的情况，并撰写短篇报告分析为什么美国注定失败，500字以内的报告"

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]}, stream_mode="values"
):
    step_num += 1

    console.print(f"\n[bold yellow]{'─' * 80}[/bold yellow]")
    console.print(f"[bold yellow]步骤 {step_num}[/bold yellow]")
    console.print(f"[bold yellow]{'─' * 80}[/bold yellow]")

    if "messages" in step:
        messages = step["messages"]

        if messages:
            msg = messages[-1]

            # 保存最终响应
            if (
                hasattr(msg, "content")
                and msg.content
                and not hasattr(msg, "tool_calls")
            ):
                final_response = msg.content  # 已经保存到了文件

            # AI 思考
            if hasattr(msg, "content") and msg.content:
                # 如果内容太长,只显示前300字符作为预览
                content = msg.content
                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    preview = content
                    console.print(
                        Panel(
                            preview,
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            content,
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green",
                        )
                    )

            # 工具调用
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_info = {
                        "工具名称": tool_call.get("name", "unknown"),
                        "参数": tool_call.get("args", {}),
                    }
                    console.print(
                        Panel(
                            JSON(json.dumps(tool_info, ensure_ascii=False)),
                            title="[bold blue]工具调用[/bold blue]",
                            border_style="blue",
                        )
                    )

            # 工具响应
            if hasattr(msg, "name") and msg.name:
                response = str(msg.content)
                console.print(
                    Panel(
                        response,
                        title=f"[bold magenta]工具响应: {msg.name}[/bold magenta]",
                        border_style="magenta",
                    )
                )

console.print("\n[bold green]任务完成![/bold green]\n")
