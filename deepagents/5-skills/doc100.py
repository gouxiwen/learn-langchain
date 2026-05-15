import os
from pathlib import Path

from dotenv import load_dotenv

# from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends import LocalShellBackend

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

root_dir = Path.cwd().as_posix()  # 转换文件分隔符为/格式而不是windows的\格式

print(root_dir)

backend = LocalShellBackend(
    root_dir=root_dir,
    inherit_env=True,
    timeout=120,  # 命令超时秒数
    max_output_bytes=100000,
)

system_prompt = r"""
## 角色设定
你是一位专业、高效、多领域的超级智能助手，具备强大的知识整合与问题解决能力。你善于理解用户意图，提供准确、清晰、有温度的回答。

## 核心任务
- 根据用户提问，结合你的专业知识库与可用工具（skills），提供高质量解答
- 回答需遵循：准确性 > 实用性 > 简洁性 > 友好性 的优先级原则
- 遇到模糊问题时，主动澄清需求；遇到复杂问题时，分步骤拆解说明

## 注意事项
read_file工具使用注意点: 不支持Windows绝对地址, 如: 错误写法 D:\xxx\xxx\SKILL.md, 正确写法为 /xxx/xxx/SKILL.md
"""

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=[root_dir + r"/skills"],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "编写 100 字左右的笑话并保存到 笑话.docx 中",
            }
        ]
    },
    config={"configurable": {"thread_id": "12345"}},
)

print(result)
