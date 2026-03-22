# 这是一个官方示例，用于演示如何使用 Deep Agents 软件包创建一个深度研究代理。
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

# 自定义指令
# 深度研究代理使用在 research_agent/prompts.py 中定义的自定义指令，这些指令是对默认中间件指令的补充（而非重复）。
# 您可以根据需要进行任意修改
from utils import format_messages
from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
# 自定义工具
# 除了内置的 DeepAgent 工具之外，深度研究代理还添加了以下自定义工具。
# 您也可以使用自己的工具，包括通过 MCP 服务器。更多详情请参阅 Deep Agents 软件包的 README 文件。
from research_agent.tools import tavily_search, think_tool

# 并发与迭代限制
max_concurrent_research_units = 3
max_researcher_iterations = 3

# 当前日期（用于提示词中的时间信息）
current_date = datetime.now().strftime("%Y-%m-%d")

# 组合主智能体的系统提示词
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# 定义研究子代理
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}

# 选择底层大模型（此处使用 Claude 4.5，Gemini 3 备选）
# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.0)
# model = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0)
env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.0
)


# 创建深度智能体
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "research context engineering approaches used to build AI agents",
            }
        ],
    }, 
)
format_messages(result["messages"])