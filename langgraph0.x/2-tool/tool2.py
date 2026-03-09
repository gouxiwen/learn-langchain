# langgraph api按照封装程度分三层
# 这里展示顶层预构建api和中间层node api调用示例，展示了如何使用langgraph预制图结构+内置工具创建一个智能体
# 内置工具调用示例
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

model = init_chat_model(
    model='glm-4.7',
    model_provider="openai",
    api_key='自己的apikey',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

search_tool = TavilySearch(
    max_results=2,
    topic='general',
    tavily_api_key='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',
)

tools = [search_tool]

search_agent = create_react_agent(model=model, tools=tools)

response = search_agent.invoke({"messages": [{"role": "user", "content": "请帮我搜索最近OpenAI CEO在访谈中的核心观点。"}]})

print(response["messages"][-1].content)