from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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


def add_tool(a: int, b: int) -> int:
    """
    This tool adds two numbers together.

    Args:
        a: The first number to add.
        b: The second number to add.

    Returns:
        The sum of the two numbers.
    """
    return a + b


def subtract_tool(a: int, b: int) -> int:
    """
    This tool subtracts two numbers together.

    Args:
        a: The first number to subtract.
        b: The second number to subtract.

    Returns:
        The difference of the two numbers.
    """
    return a - b


def divide_tool(a: int, b: int) -> float:
    """
    This tool divides two numbers together.

    Args:
        a: The first number to divide.
        b: The second number to divide.

    Returns:
        The quotient of the two numbers.
    """
    return a / b


tools = [multiply_tool, add_tool, subtract_tool, divide_tool]

SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
chat_model_with_tools = chat_model.bind_tools(tools)


def assistant(state: ChatWithToolsState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    return {"messages": [chat_model_with_tools.invoke(messages)]}


memory = MemorySaver()
graph_builder = StateGraph(ChatWithToolsState)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "assistant")
graph_builder.add_conditional_edges("assistant", tools_condition)
graph_builder.add_edge("tools", "assistant")

chat_model_graph = graph_builder.compile(checkpointer=memory)
# messages = chat_model_graph.invoke({"messages": HumanMessage(content="Hello!")})


input_messages = [
    HumanMessage(
        content="Add 3 and 4. Multiply the output by 2. Divide the output by 4"
    )
]
config = {"configurable": {"thread_id": "123"}}
messages = chat_model_graph.invoke({"messages": input_messages}, config=config)
for m in messages["messages"]:
    m.pretty_print()


input_messages = [HumanMessage(content="Convert that to int and then multiply by 2")]
messages = chat_model_graph.invoke({"messages": input_messages}, config=config)
for m in messages["messages"]:
    m.pretty_print()
