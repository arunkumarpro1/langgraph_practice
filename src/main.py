from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

gpt41_mini_chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


def check_chat():
    # Create a message
    msg = HumanMessage(content="Hello world", name="Lance")
    # Message list
    messages = [msg]
    # Invoke the model with a list of messages
    response = gpt4o_chat.invoke(messages)
    print(response)


def check_search():
    tavily_search = TavilySearch(max_results=3)
    search_docs = tavily_search.invoke("What is LangGraph?")
    print(search_docs["results"])


def main():
    # check_chat()
    check_search()


if __name__ == "__main__":
    main()
