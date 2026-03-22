# deepagent运行背后的中间件-任务清单中间件
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from deepagents import FilesystemMiddleware
from langchain.messages import HumanMessage
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
# backend（可选）：该参数决定记忆的存储模式，是 FileSystem 中间件的核心配置，后续章节将重点展开讲解。
# system_prompt（可选）：允许开发者重写默认的系统提示词，以引导 Agent 在何时使用文件操作。
# custom_tool_descriptions（可选）：可用于自定义各个工具的描述文本，以适应特定场景的需求。
# agent = create_agent(
#     model=model,
#     middlewares=[
#         FilesystemMiddleware(
#             backend=None,  # Optional: custom backend (defaults to StateBackend)
#             system_prompt="Write to the filesystem when...",  # Optional custom addition to the system prompt
#             custom_tool_descriptions={
#                 "ls": "Use the ls tool when...",
#                 "read_file": "Use the read_file tool to...",
#             },  # Optional: Custom descriptions for filesystem tools
#         ),
#     ],
# )


# 不同backend示例
# FileSystemBackend,访问本地磁盘
def create_filesystem_backend():
    from deepagents.backends import FilesystemBackend

    agent_local = create_agent(
        model=model,
        tools=[],
        middleware=[
            FilesystemMiddleware(
                backend=FilesystemBackend(root_dir="./test_dir", virtual_mode=True)
            )
        ],
    )

    res = agent_local.invoke(
        {
            "messages": [
                HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '测试'")
            ]
        }
    )

    print(res)


# StateBackend：线程级短期记忆
def create_state_backend():
    from deepagents.backends import StateBackend

    agent_local = create_agent(
        model=model,
        tools=[],
        middleware=[
            FilesystemMiddleware(backend=lambda runtime: StateBackend(runtime))
        ],
    )

    res = agent_local.invoke(
        {
            "messages": [
                HumanMessage(
                    "调用工具写入一个文件，文件名为:测试.txt, 内容为: '你好帅'"
                ),
                HumanMessage("调用工具读取名为测试.txt的文件，告诉我里面的内容"),
            ]
        },
    )

    print(res["messages"][-1].content)


# StoreBackend：跨线程长期记忆
def create_store_backend():
    from deepagents.backends import StoreBackend
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    agent_local1 = create_agent(
        model=model,
        tools=[],
        store=store,
        middleware=[
            FilesystemMiddleware(backend=lambda runtime: StoreBackend(runtime))
        ],
    )

    agent_local1.invoke(
        {
            "messages": [
                HumanMessage(
                    "调用工具写入一个文件，文件名为:测试.txt, 内容为: '你好帅'"
                ),
            ]
        }
    )

    agent_local2 = create_agent(
        model=model,
        tools=[],
        store=store,  # 同一个store实例
        middleware=[
            FilesystemMiddleware(backend=lambda runtime: StoreBackend(runtime))
        ],
    )

    res = agent_local2.invoke(
        {
            "messages": [
                HumanMessage("调用工具读取名为测试.txt的文件，告诉我里面的内容")
            ]
        },
    )

    print(res["messages"][-1].content)


# CompositeBackend：复合后端（混合存储）
def create_composite_backend():
    from deepagents import FilesystemMiddleware
    from deepagents.backends import (
        StateBackend,
        StoreBackend,
        CompositeBackend,
    )
    from langchain.messages import HumanMessage
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    composite_backend = lambda runtime: CompositeBackend(
        default=StateBackend(runtime), routes={"/memories/": StoreBackend(runtime)}
    )

    agent = create_agent(
        model=model,
        store=store,
        middleware=[FilesystemMiddleware(backend=composite_backend)],
    )

    config1 = {"configurable": {"thread_id": "1"}}

    # 智能体将 "preferences.txt" 写入 /memories/ 路径
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我最爱的水果是草莓, 请把我的偏好保存在/memories/preferences.txt",
                }
            ]
        },
        config=config1,
    )

    config2 = {"configurable": {"thread_id": "2"}}

    res = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "请从/memories/获取我最爱的水果是什么?"}
            ]
        },
        config=config2,
    )

    print(res["messages"][-1].content)


if __name__ == "__main__":
    # create_filesystem_backend()
    # create_state_backend()
    # create_store_backend()
    create_composite_backend()
