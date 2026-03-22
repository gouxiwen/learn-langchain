# deepagent运行背后的中间件-任务清单中间件
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

# from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model="glm-5",
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


# 配置任务清单中间件，并创建一个包含该中间件的代理。
agent = create_agent(
    model=model,
    tools=[],
    middleware=[TodoListMiddleware()],
)

# 通过复杂案例测试： 设计了一个高度复杂的、需要多步分析和计算的综合性问题来测试。当用户下达这个任务后，Todo List机制会自动生效。
# 在Agent的最终响应中，除了常规的messages（包含对话历史和最终答案），还会发现一个新增的todos字段。这个字段中包含了Agent对原始任务进行拆解后的详细子任务列表。每个子任务条目都包含content（任务描述）和status（状态）。
# 子任务的状态共有三种：
# pending：尚未执行的子任务。
# progress：当前正在执行的子任务。
# completed：已经完成的子任务。
status = {
    "messages": [
        """你要一步一步的详细规划以下内容再进行回答。
请分析美国加利福尼亚中央谷地的杏仁种植业在未来30年面临的气候变化风险,并估算其经济影响。
具体需要回答,:
“假设当前气候趋势持续,到2050年,加利福尼亚中央谷地杏仁产量可能减少的百分比及其对该州经
济的潜在年度损失是多少美元?这些美元按照2025年11月的汇率能够购买多少比特币?
"""
    ]
}

result = agent.invoke(status)
print(result)
print("\n---------TODO---------------\n")
print(result["todos"])
