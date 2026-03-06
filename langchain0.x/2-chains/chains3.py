from langchain_classic.chat_models import init_chat_model
from langchain_classic.prompts import PromptTemplate # 导入聊天提示模板组件
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser # 导入结构化输出组件

schemas = [ # 构建结构化数据模板
    ResponseSchema(name="name", description="用户的姓名"),
    ResponseSchema(name="age", description="用户的年龄")
]

parser = StructuredOutputParser.from_response_schemas(schemas) # 根据模板生成解析器

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-vcsmjxxaanktozmkjkqkipjsisvbcoezstlvsxslbqqfddir", #你注册的硅基流动api_key
)

prompt = PromptTemplate.from_template(
    "请根据以下内容提取用户信息，并返回 JSON 格式：\n{input}\n\n{format_instructions}"
) # 这是另一种使用占位符的提示词模板表示方式

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions()) 
    | model
    | parser
)

result = chain.invoke({"input": "用户叫李雷，今年25岁，是一名工程师。"}) # 输入input, format_instructions前面已经赋值

print(result)