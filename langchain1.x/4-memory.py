import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

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
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {
        "messages": "你好我叫吴彦祖?"
    },
    {
        "configurable": {
            "thread_id": "1"
        }
    }
)

for msg in result['messages']:
    msg.pretty_print()

result = agent.invoke(
    {
        "messages": "你好我叫什么名字?"
    },
{
        "configurable": {
            "thread_id": "1"
        }
    }
)

for msg in result['messages']:
    msg.pretty_print()