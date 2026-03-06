from langchain_classic.chat_models import init_chat_model

def siliconflow():
    model = init_chat_model(
        model="Qwen/Qwen3-8B", # 模型名称
        model_provider="openai", # 模型提供商，硅基流动提供了openai请求格式的访问
        base_url="https://api.siliconflow.cn/v1/", #硅基流动模型的请求url
        api_key="sk-vcsmjxxaanktozmkjkqkipjsisvbcoezstlvsxslbqqfddir", # 填写你注册的硅基流动 API Key
    )

    question = "你好，请介绍一下你自己"

    result = model.invoke(question) #将question问题传递给model组件, 同步调用大模型生成结果

    print(result)
    print(type(result))

def deepseek():
    model = init_chat_model(
        model='deepseek-chat', # deepseek-chat表示调用DeepSeek-v3模型，deepseek-reasoner表示调用DeepSeek-R1模型，
        model_provider='deepseek',# 模型提供商写deepseek
        api_key="sk-77e0e8b92d8e46f5ad6c5ff7324fd0ed", #你注册的deepseek api_key
    )

    question="你好，请介绍一下你自己"

    result = model.invoke(question)
    print(result)
    print(type(result))

if __name__ == "__main__":
    siliconflow()
    # deepseek()