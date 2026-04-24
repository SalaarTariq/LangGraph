from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Tools
tavily_search = TavilySearchResults(max_results=3)
wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

tools = [tavily_search, wikipedia_search]
llm_with_tools = llm.bind_tools(tools)


# State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ReAct System Prompt
SYSTEM_PROMPT = (
    "You are a ReAct agent. For every user question, follow this loop:\n"
    "1. Thought: Think about what information you need.\n"
    "2. Action: Call a tool to get that information.\n"
    "3. Observation: Read the tool result.\n"
    "4. Repeat if needed, then give a final answer.\n\n"
    "Tools:\n"
    "- tavily_search_results_json: Real-time search (news, current events, dates).\n"
    "- wikipedia: Factual knowledge (history, biographies, science).\n\n"
    "You must call at least one tool before answering."
)


# Nodes
def agent(state: State) -> State:
    response = llm_with_tools.invoke(
        [("system", SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


# Graph: ReAct loop
#
# START -> agent -> (tools_condition) -> tools -> agent -> ... -> END
#                                     -> END (if no tool call)

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()

# Run
result = app.invoke({
    "messages": [("user", "What were the main findings of the Oscars?")]
})

print(result["messages"][-1].content)
