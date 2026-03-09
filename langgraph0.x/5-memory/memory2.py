# 短记忆持久化
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver

env_path = Path(__file__).parent.parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 第一次调用时必须要setup()


    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}


    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    response = graph.invoke(
        {"messages": [{"role": "user", "content": "你好，我是苍老师"}]},
        config
    )

    print(response['message'])

    response = graph.invoke(
        {"messages": [{"role": "user", "content": "请问我叫什么名字"}]},
        config
    )

    print(response['message'])