from langchain_classic.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser # 导入标准输出组件


model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-vcsmjxxaanktozmkjkqkipjsisvbcoezstlvsxslbqqfddir", #你注册的硅基流动api_key
)

# 搭建链条，把model和字符串输出解析器组件连接在一起
basic_qa_chain =  model | StrOutputParser()
question = "你好，请你介绍一下你自己。"
result = basic_qa_chain.invoke(question)

print(result)