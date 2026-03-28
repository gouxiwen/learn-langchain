# langgraph使用Command对象：边定义的第二种方法，这里演示一下
# 还有memory，中断，恢复api都适用
from langgraph.graph import START, END, StateGraph
import operator
from typing import TypedDict, List, Annotated, Literal
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    nList: Annotated[List[str], operator.add]


def node_a(state: State) -> Command[Literal["b", "c", END]]:
    print("进入 A 节点")
    select = state["nList"][-1]
    if select == "b":
        next_node = "b"
    elif select == "c":
        next_node = "c"
    elif select == "q":
        next_node = END
    else:
        admin = interrupt(f"未期望的输出 {select}")
        print("用户重新输入是:", admin)
        if admin == "continue":
            next_node = "b"
            select = "b"
        else:
            next_node = END
            select = "q"

    return Command(update=State(nList=[select]), goto=next_node)


def node_b(state: State):
    return Command(goto=END)


def node_c(state: State):
    return Command(goto=END)


# 构建图
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_edge(START, "a")

# 配置内存检查点器和线程
memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

# 编译时传入检查点器
graph = builder.compile(checkpointer=memory)

# 实现外部中断处理循环:
while True:
    user = input("b, c or q to quit:")
    input_state = State(nList=[user])
    result = graph.invoke(input_state, config)
    print(result)

    if "__interrupt__" in result:
        print(f"Interrupt:{result}")
        msg = result["__interrupt__"][-1].value
        print(msg)
        human = input(f"\n{msg}, 重新输入: ")
        human_response = Command(resume=human)
        result = graph.invoke(human_response, config)

    if result["nList"][-1] == "q":
        print("quit")
        break
