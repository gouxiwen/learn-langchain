# LangChain对于Agent的定义是“由大模型规划并自由组装各种链来满足用户需求”
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.chat_models import init_chat_model

# 初始化 Playwright 浏览器：
sync_browser = create_sync_playwright_browser() # 创建同步执行的浏览器
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser) # 构建PlayWright浏览器工具
tools = toolkit.get_tools() # 获取PlayWright浏览器操作函数

# 初始化模型
# 使用 智普 模型
model = init_chat_model(
    # model="glm-5",
    model="glm-4.7",
    model_provider="openai", # 模型提供商，智普提供了openai请求格式的访问
    openai_api_key="75301e9d6ffc4d878a32a2a5b31dc8c0.frRvWZTAQklAYIXJ",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 通过 LangChain Hub 拉取提示词模版
prompt = hub.pull("hwchase17/openai-tools-agent")

# 拉取提示词模板的代码与以下自定义提示词的代码等价:
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant"),
#         ('placeholder': "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),  # 这部分agnet提示符不需要人工输入，同时也是写死的不可以修改
#     ]
# )

# 通过 LangChain 创建 OpenAI 工具代理
agent = create_openai_tools_agent(model, tools, prompt)

# 通过 AgentExecutor 执行代理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 定义任务
    command = {
        "input": "访问这个网站 https://www.microsoft.com/en-us/microsoft-365/blog/2025/01/16/copilot-is-now-included-in-microsoft-365-personal-and-family/?culture=zh-cn&country=cn 并帮我总结一下这个网站的内容"
    }

    # 执行任务
    response = agent_executor.invoke(command)
    print(response)