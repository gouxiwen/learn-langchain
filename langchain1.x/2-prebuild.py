# 使用create_agent创建一个Agent,提示词，大模型和工具函数三要素必不可少
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def get_weather(loc:str)->str:
    """
    根据地点参数可以返回该地点的天气情况
    """
    return f"{loc} 天气是晴！气温23°"

SYSTEM_PROMPT = "你是一个天气助手，具备调用get_weather天气函数获取指定地点天气的能力"

env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT
)

question="北京的天气怎么样?"

# 使用 agent.stream 方法实现流式输出
# 通过 pretty_print() 方法格式化显示
for step in agent.stream(
    {'messages': question},
    # stream_mode="values"
    stream_mode="messages" # 逐个token输出
    # stream_mode="custom"
    ):
    step["messages"][-1].pretty_print()