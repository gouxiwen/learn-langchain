# 长记忆
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

store = InMemoryStore()

store.put(
    ("users",),
    "user_123",
    {
        "name": "苍老师",
        "language": "日语",
    }
)

def get_user_info(config: RunnableConfig) -> str:
    """查找用户信息的函数，可以查看长期记忆中储存的用户信息"""
    # Same as that provided to `create_react_agent`
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"



agent = create_react_agent(
    model=model,
    tools=[get_user_info],
    store=store
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查找长期记忆中储存的用户信息"}]},
    config={"configurable": {"user_id": "user_123"}}
)

print(response['messages'])