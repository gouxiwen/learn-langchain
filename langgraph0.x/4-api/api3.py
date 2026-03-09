# langgraph api按照封装程度分三层
# 这里展示底层 api调用示例
# 多步骤中的状态传递与条件控制

from langgraph.graph import StateGraph,START, END
from pydantic import BaseModel
from typing import Optional

class MyState(BaseModel):
    x: int
    result: Optional[str] = None

builder = StateGraph(MyState)

# 定义各节点处理逻辑（接受 MyState，返回 MyState）
def check_x(state: MyState) -> MyState:
    print(f"[check_x] Received state: {state}")
    return state

def is_even(state: MyState) -> bool:
    return state.x % 2 == 0

def handle_even(state: MyState) -> MyState:
    print("[handle_even] x 是偶数")
    return MyState(x=state.x, result="even")

def handle_odd(state: MyState) -> MyState:
    print("[handle_odd] x 是奇数")
    return MyState(x=state.x, result="odd")

builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)

# 添加条件分支
builder.add_conditional_edges("check_x", is_even, {
    True: "handle_even",
    False: "handle_odd"
})

# 衔接起始和结束
builder.add_edge(START, "check_x")
builder.add_edge("handle_even", END)
builder.add_edge("handle_odd", END)

# 编译图
graph = builder.compile()

# 执行测试
print("\n测试 x=4（偶数）")
graph.invoke(MyState(x=4))

print("\n测试 x=3（奇数）")
graph.invoke(MyState(x=3))
