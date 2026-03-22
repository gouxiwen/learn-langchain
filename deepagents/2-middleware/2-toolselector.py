# deepagent运行背后的中间件-工具选择器
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

# from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
# glm-4.7｜glm-5｜glm-5-turbo都无法使用LLMToolSelectorMiddleware，即智谱模型不支持wrap_model_call‌钩子，智谱有自己的官方工具ZhipuAIAllToolsRunnable，作用类似
# 所以可以使用deepseek-chat或qwen3选择工具
model = init_chat_model(
    model="glm-5",
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

# model_deepseek = init_chat_model(
#     model="deepseek-chat",  # deepseek-chat表示调用DeepSeek-v3模型，deepseek-reasoner表示调用DeepSeek-R1模型，
#     model_provider="deepseek",  # 模型提供商写deepseek
#     api_key=os.getenv("deepseek_api_key"),
# )

model_qwen3 = init_chat_model(
    model="Qwen/Qwen3-8B",  # 模型名称
    model_provider="openai",  # 模型提供商，硅基流动提供了openai请求格式的访问
    base_url="https://api.siliconflow.cn/v1/",  # 硅基流动模型的请求url
    api_key=os.getenv("siliconflow_api_key"),  # 填写你注册的硅基流动 API Key
)


# 编写一个核心的 calculate 工具，并同时定义 tool_1 至 tool_4 作为备用工具，以模拟一个拥有较多工具的场景。
@tool
def tool_1(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_2(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_3(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def tool_4(input: str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations and return the result.
    Args:
        expression: Mathematical expression to evaluate
        (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")
    Returns:
        The calculated result as a string
    """
    result = str(eval(expression))
    return result


# 配置工具选择器中间件，并创建一个包含该中间件的代理。
# model: 负责执行工具筛选的模型实例，本例中与主模型共用同一个model。
# max_tools: 每次为主模型筛选出的最大工具数量。
# always_include: 一个列表，用于指定无论如何都必须被选中的工具名称。
# 需要注意的是，create_agent的tools参数传入的是工具对象列表，而always_include中传入的是这些工具对应的名称字符串。
agent = create_agent(
    model=model,
    tools=[tool_1, tool_2, tool_3, tool_4, calculate],
    middleware=[
        LLMToolSelectorMiddleware(
            model=model,
            max_tools=2,
            always_include=["tool_1"],
        ),
    ],
)

# 运行测试： 最后通过一个数学计算任务来测试效果。
# 从运行结果可以看到，尽管工具列表庞大，但智能体依然能够准确地选择并使用 calculate 工具来完成计算，这证明了Tool Selector在筛选工具方面的有效性。
status = {"messages": "请计算2+3*4的值"}

result = agent.invoke(status)
print(result)
