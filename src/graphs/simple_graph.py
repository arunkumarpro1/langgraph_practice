from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class SimpleGraphState(BaseModel):
    graph_state: str


graph_builder = StateGraph(SimpleGraphState)


def node_1(state: SimpleGraphState) -> SimpleGraphState:
    return SimpleGraphState(graph_state=state.graph_state + " I'm ")


def node_2(state: SimpleGraphState) -> SimpleGraphState:
    return SimpleGraphState(graph_state=state.graph_state + "happy")


def node_3(state: SimpleGraphState) -> SimpleGraphState:
    return SimpleGraphState(graph_state=state.graph_state + "sad")


def decide_mood(state: SimpleGraphState) -> Literal["node_2", "node_3"]:
    import random

    if random.random() < 0.5:
        return "node_2"
    else:
        return "node_3"


graph_builder.add_node("node_1", node_1)
graph_builder.add_node("node_2", node_2)
graph_builder.add_node("node_3", node_3)

graph_builder.add_edge(START, "node_1")
graph_builder.add_conditional_edges("node_1", decide_mood)
graph_builder.add_edge("node_2", END)
graph_builder.add_edge("node_3", END)

graph = graph_builder.compile()
# display(graph.get_graph().draw_mermaid_png())

response = graph.invoke({"graph_state": "Hello, I am Arun."})
print(response)
