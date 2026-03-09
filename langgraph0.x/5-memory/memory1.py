# 短记忆
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

checkpointer = InMemorySaver() # 设置检查点


agent = create_react_agent(model=model,
                           tools=[],
                           checkpointer=checkpointer)

#  短期记忆与线程相关，在与智能体对话时需要携带config线程id信息
config = {
    "configurable": {
        "thread_id": "1"
    }
}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫苍老师，好久不见！"}]},
    config
)

print(response['messages'][-1].content)


response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请问你还记得我叫什么名字么？"}]},
    config
)

print('------------线程1------------------')
print(response['messages'][-1].content)

new_config = {
    "configurable": {
        "thread_id": "2"
    }
}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请问你还记得我叫什么名字么？"}]},
    new_config
)

print('------------线程2------------------')
print(response['messages'][-1].content)