import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

# from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from deepagents.backends import FilesystemBackend

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

research_instruction = """
 你是一位从事国际关系研究的专家，能够分析不同国家的国情，并按照用户的要求生成报告
"""

agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=research_instruction,
    subagents=[],
    backend=FilesystemBackend(root_dir="./test_dir", virtual_mode=True),
)

query = "请分析伊朗和美国的国情，并帮我撰写一份1500字左右的为什么伊朗和美国会对立的报告"
step_num = 0
final_response = None

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
