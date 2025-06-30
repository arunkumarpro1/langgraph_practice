from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class ChatWithToolsState(MessagesState):
    pass


def multiply_tool(a: int, b: int) -> int:
    """
    This tool multiplies two numbers together.

    Args:
        a: The first number to multiply.
        b: The second number to multiply.

    Returns:
        The product of the two numbers.
    """
    return a * b


chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
chat_model_with_tools = chat_model.bind_tools([multiply_tool])


def tool_calling_chat_model(state: ChatWithToolsState):
    return {"messages": [chat_model_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(ChatWithToolsState)
graph_builder.add_node("tool_calling_chat_model", tool_calling_chat_model)
graph_builder.add_node("tools", ToolNode([multiply_tool]))
graph_builder.add_edge(START, "tool_calling_chat_model")
graph_builder.add_conditional_edges("tool_calling_chat_model", tools_condition)
graph_builder.add_edge("tools", "tool_calling_chat_model")

chat_model_graph = graph_builder.compile()

config = {"configurable": {"thread_id": "1"}}

messages = chat_model_graph.invoke(
    {"messages": [HumanMessage(content="Multiply 243 and 3")]}, config=config
)

for m in messages["messages"]:
    m.pretty_print()
