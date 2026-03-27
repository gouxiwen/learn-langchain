# 自定义中间件实现模型选择
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model


@dataclass
class Context:
    user_level: str = "expert"


env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
zhipu_model = init_chat_model(
    model="glm-4.7",
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

Qwen3_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("siliconflow_api_key"),
)


class ExpertiseBasedToolMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        user_level = request.runtime.context.user_level

        if user_level == "expert":
            model = zhipu_model
            tools = []
        else:
            # Less powerful model
            model = Qwen3_model
            tools = []

        request.model = model
        request.tools = tools
        return handler(request)


agent = create_agent(
    model=Qwen3_model,
    tools=[],
    middleware=[ExpertiseBasedToolMiddleware()],
    context_schema=Context,
)


question = "你好请问你是?"

# 专家用户场景
for step in agent.stream(
    {"messages": {"role": "user", "content": question}},
    context=Context(user_level="expert"),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# 普通用户场景
for step in agent.stream(
    {"messages": {"role": "user", "content": question}},
    context=Context(user_level="student"),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
