from langchain_classic.chat_models import init_chat_model
from langchain_classic.prompts import ChatPromptTemplate # 导入聊天提示模板组件
from langchain_classic.output_parsers import BooleanOutputParser # 导入布尔输出组件

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-vcsmjxxaanktozmkjkqkipjsisvbcoezstlvsxslbqqfddir", #你注册的硅基流动api_key
)

prompt_template = ChatPromptTemplate.from_messages([
     ("system", "你是一个乐意助人的助手，请根据用户的问题给出回答"),
    ("user", "这是用户的问题： {topic}， 请用 yes 或 no 来回答")
])

# 搭建链条，把model和字符串输出解析器组件连接在一起
bool_qa_chain = prompt_template | model | BooleanOutputParser()
question = "请问 1 + 1 是否 大于 2？"
result = bool_qa_chain.invoke({'topic':question})


print(result)