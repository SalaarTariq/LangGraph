from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    platform: str


def router(state: State) -> State:
    last_msg = state["messages"][-1].content

    response = llm.invoke(
        f"User message: '{last_msg}'\n\n"
        "Your job: check if the user mentioned a specific platform (facebook, instagram, twitter).\n"
        "Examples:\n"
        "- 'write a tweet about AI' -> twitter\n"
        "- 'make an instagram post about food' -> instagram\n"
        "- 'create a facebook post about tech' -> facebook\n"
        "- 'write a post about AI' -> all\n"
        "- 'make it shorter' -> all\n"
        "- 'add more hashtags' -> all\n\n"
        "Reply with ONLY one word: facebook, instagram, twitter, or all."
    )
    platform = response.content.strip().lower()
    if platform not in ("facebook", "instagram", "twitter", "all"):
        platform = "all"
    print(f"\nRouter -> {platform}")
    return {"platform": platform}


def route_to_platform(state: State) -> list[str]:
    if state["platform"] == "all":
        return ["facebook", "instagram", "twitter"]
    return [state["platform"]]


def facebook_node(state: State) -> State:
    response = llm.invoke([
        ("system",
         "You are a Facebook post writer. Write casual, engaging Facebook posts.\n"
         "If the user asks for changes to a previous post, look at the conversation history and modify accordingly."),
    ] + state["messages"])
    return {"messages": [("assistant", f"[FACEBOOK]\n{response.content}")]}


def instagram_node(state: State) -> State:
    response = llm.invoke([
        ("system",
         "You are an Instagram caption writer. Use emojis and hashtags.\n"
         "If the user asks for changes to a previous post, look at the conversation history and modify accordingly."),
    ] + state["messages"])
    return {"messages": [("assistant", f"[INSTAGRAM]\n{response.content}")]}


def twitter_node(state: State) -> State:
    response = llm.invoke([
        ("system",
         "You are a Twitter/X post writer. Keep tweets under 280 characters.\n"
         "If the user asks for changes to a previous post, look at the conversation history and modify accordingly."),
    ] + state["messages"])
    return {"messages": [("assistant", f"[TWITTER]\n{response.content}")]}


# Build graph
graph = StateGraph(State)

graph.add_node("router", router)
graph.add_node("facebook", facebook_node)
graph.add_node("instagram", instagram_node)
graph.add_node("twitter", twitter_node)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_to_platform, ["facebook", "instagram", "twitter"])
graph.add_edge("facebook", END)
graph.add_edge("instagram", END)
graph.add_edge("twitter", END)

# Memory: MemorySaver persists messages across invocations
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# thread_id keeps conversation in the same memory thread
config = {"configurable": {"thread_id": "1"}}

print("Social Media Post Agent (type 'quit' to exit)")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "quit":
        break

    result = app.invoke(
        {"messages": [("user", user_input)]},
        config=config,
    )

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content.startswith("["):
            print(f"\n{msg.content}")
