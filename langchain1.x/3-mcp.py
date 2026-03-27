# 调用mcp和结构化输出
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy
from pydantic import BaseModel

class Result(BaseModel):
    loc1: str
    loc2: str
    distance: float

env_path = Path(__file__).parent.parent.joinpath(".env.local")
load_dotenv(dotenv_path=env_path, override=True)
model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key=os.getenv("zhipu_api_key"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

mcp_client = MultiServerMCPClient(
    {
        "amap-maps": {
              "command": "cmd",
              "args": [
                "/c",
                "npx",
                "-y",
                "@amap/amap-maps-mcp-server"
              ],
              "env": {
                "AMAP_MAPS_API_KEY": os.getenv("amap_api_key")
              },
              'transport': 'stdio'
            }
    }
)

async def get_server_tools():
    mcp_tools = await mcp_client.get_tools()
    print(f"加载了{len(mcp_tools)}: {[t.name for t in mcp_tools]}")
    agent_with_mcp = create_agent(
        model,
        tools=mcp_tools,
        system_prompt = "你是一个高德地图规划助手，能帮我规划形成和获得地图基本信息",
        response_format=AutoStrategy(Result), # 结构化输出
        
    )
    result = await agent_with_mcp.ainvoke(
        {
            "messages":{
                "role": 'user',
                "content": "请告诉我北京圆明园到北京西北旺地铁站距离"
            }
        }
    )
    for msg in result['messages']:
        msg.pretty_print()


asyncio.run(get_server_tools())