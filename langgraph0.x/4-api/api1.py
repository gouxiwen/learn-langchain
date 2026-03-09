# langgraph api按照封装程度分三层
# 这里展示底层 api调用示例
# 基本使用state，node，edge构建图结构，并执行图结构得到结果

from langgraph.graph import StateGraph,START, END

# 创建一个字典类型的State
builder = StateGraph(dict)

# 定义两个函数，分别表示加法和减法操作
def addition(state):
    print(f'加法节点收到的初始值:{state}')
    return {"x": state["x"] + 1}

def subtraction(state):
    print(f'减法节点收到的初始值:{state}')
    return {"x": state["x"] - 2}


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
initial_state={"x": 10}

result = graph.invoke(initial_state)

print('最后的结果:', result)

