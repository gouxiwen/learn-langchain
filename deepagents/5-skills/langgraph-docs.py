import os
from pathlib import Path

from dotenv import load_dotenv

# from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
# model = init_chat_model(
#     model="Qwen/Qwen3-8B",  # 模型名称
#     model_provider="openai",  # 模型提供商，硅基流动提供了openai请求格式的访问
#     base_url="https://api.siliconflow.cn/v1/",  # 硅基流动模型的请求url
#     api_key=os.getenv("siliconflow_api_key"),  # 填写你注册的硅基流动 API Key
#     temperature=0.0,
# )
model = init_chat_model(
    model="deepseek-chat",  # deepseek-chat表示调用DeepSeek-v3模型，deepseek-reasoner表示调用DeepSeek-R1模型，
    model_provider="deepseek",  # 模型提供商写deepseek
    api_key=os.getenv("deepseek_api_key"),
    temperature=0.0,
)

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir="./", virtual_mode=True),
    skills=["./skills/"],
    checkpointer=checkpointer,  # Required!
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is langgraph?",
            }
        ]
    },
    config={"configurable": {"thread_id": "12345"}},
)

print(result)
