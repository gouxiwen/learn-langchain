# LangChain对于Agent的定义是“由大模型规划并自由组装各种链来满足用户需求”
from langchain_classic.agents import create_tool_calling_agent, tool, AgentExecutor
from langchain_classic.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

# 搜索工具
# Tavily每个月有1000次免费搜索案例
# api key在环境变量TAVILY_API_KEY中，或者直接在代码中传入
search = TavilySearch(max_results=2, tavily_api_key='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')
tools = [search]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名助人为乐的助手，并且可以调用工具进行网络搜索，获取实时信息。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # 这部分agnet提示符不需要人工输入，同时也是写死的不可以修改
    ]
)
# 初始化模型
# 使用 智普 模型
model = init_chat_model(
    # model="glm-5",
    model="glm-4.7",
    model_provider="openai", # 模型提供商，智普提供了openai请求格式的访问
    api_key='自己的apikey',
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "请问苹果2025WWDC发布会召开的时间是？"})
print(response)
