# langgraph1.x的基本核心概念，state，node,edge都和0.x版本一样，langgraph0.x版本就有reducer，这里特别学习一下
# 当一个节点拥有多个出边（即触发了并行执行）时，多个后续节点可能同时更新 State 中的同一个字段。此时就会产生冲突：以谁的更新为准？
# 为了更精确的控制，这时就需要用到Reducer（归约器）来解决解决此问题。
# 它是一个合并函数，定义了当多个更新同时作用于同一个状态字段时，应如何合并这些更新
from typing import TypedDict, List, Annotated
from langgraph.graph import START, END, StateGraph


def deduplicate_merge(old_list: List[str], new_list: List[str]) -> List[str]:
    """自定义Reducer：合并列表并去重"""
    combined = old_list + new_list
    return list(dict.fromkeys(combined))  # 保持顺序的去重


class State(TypedDict):
    unique_items: Annotated[List[str], deduplicate_merge]


def node_a(state: State) -> State:
    print(f"Adding 'A' to {state['unique_items']}")
    return State(unique_items=["A"])


def node_A_extra(state: State) -> State:
    print(f"Adding 'A' to {state['unique_items']}")
    return State(unique_items=["A"])


builder = StateGraph(State)

builder.add_node("a", node_a)
builder.add_node("a_extra", node_A_extra)

builder.add_edge(START, "a")
builder.add_edge("a", "a_extra")
builder.add_edge("a_extra", END)

graph = builder.compile()

initial_state = State(unique_items=["Initial String"])

print(graph.invoke(initial_state))
# 上面代码的图结构如下所示，按理说列表中应该有两个“A”，但是因为添加时会去重，所以最后列表中只有这一个“A”了。


# 条件边
# def conditional_edge(state: State) -> Literal['b', 'c', END]:
#     select = state["nList"][-1]
#     if select == "b":
#         return 'b'
#     elif select == 'c':
#         return 'c'
#     elif select == 'q':
#         return END
#     else:
#         return END

# builder.add_conditional_edges("a", conditional_edge)
