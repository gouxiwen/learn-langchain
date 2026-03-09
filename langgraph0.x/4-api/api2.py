# langgraph api按照封装程度分三层
# 这里展示底层 api调用示例
# 约束state类型

from langgraph.graph import StateGraph,START, END
from pydantic import BaseModel

# 1. 定义结构化状态模型
class CalcState(BaseModel):
    x: int

# 2. 定义节点函数，接收并返回 CalcState
def addition(state: CalcState) -> CalcState:
    print(f"[addition] 初始状态: {state}")
    return CalcState(x=state.x + 1)

def subtraction(state: CalcState) -> CalcState:
    print(f"[subtraction] 接收到状态: {state}")
    return CalcState(x=state.x - 2)

# 创建一个CalcState类型的State
builder = StateGraph(CalcState)

# 向图中添加两个节点
builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

# 构建节点之间的边
builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

# 执行图的编译，需要通过调用compile()方法将编排后的builder编译成一个可执行的图
graph = builder.compile()

# 打印可视化效果
graph.get_graph().print_ascii()

# 执行图，invoke()方法接受一个初始状态作为输入，并返回最终的结果
initial_state=CalcState(x=10)

result = graph.invoke(initial_state)

print('最后的结果:', result)

