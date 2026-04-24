from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

tavily_search = TavilySearchResults(max_results=3)
wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

tools = [tavily_search, wikipedia_search]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> State:
    response = llm_with_tools.invoke([
        ("system",
         "You are a helpful research assistant with access to two tools:\n"
         "1. tavily_search_results_json - Use for real-time, current information like news, today's date, weather, recent events.\n"
         "2. wikipedia - Use for factual knowledge, history, definitions, biographies, science topics.\n"
         "Choose the right tool based on the user's question. Always use a tool before answering."),
    ] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

app = graph.compile()

print("=== Query 1: Real-time (should use Tavily) ===")
result = app.invoke({"messages": [("user", "What is today's date in Pakistan?")]})
print(result["messages"][-1].content)

print("\n=== Query 2: Knowledge (should use Wikipedia) ===")
result = app.invoke({"messages": [("user", "Who was Albert Einstein and what did he discover?")]})
print(result["messages"][-1].content)
