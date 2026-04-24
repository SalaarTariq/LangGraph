from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    topic: str
    facebook_post: str
    instagram_post: str
    twitter_post: str


def facebook_node(state: State) -> State:
    response = llm.invoke(f"Write a Facebook post about: {state['topic']}. Keep it casual and engaging.")
    return {"facebook_post": response.content}


def instagram_node(state: State) -> State:
    response = llm.invoke(f"Write an Instagram caption about: {state['topic']}. Use emojis and hashtags.")
    return {"instagram_post": response.content}


def twitter_node(state: State) -> State:
    response = llm.invoke(f"Write a tweet about: {state['topic']}. Keep it under 280 characters.")
    return {"twitter_post": response.content}


graph = StateGraph(State)

graph.add_node("facebook", facebook_node)
graph.add_node("instagram", instagram_node)
graph.add_node("twitter", twitter_node)

# Fan-out: START -> all 3 nodes in parallel
graph.add_edge(START, "facebook")
graph.add_edge(START, "instagram")
graph.add_edge(START, "twitter")

# Fan-in: all 3 nodes -> END
graph.add_edge("facebook", END)
graph.add_edge("instagram", END)
graph.add_edge("twitter", END)

app = graph.compile()

result = app.invoke({"topic": "Importance of AI in 2026", "facebook_post": "", "instagram_post": "", "twitter_post": ""})

print("=== Facebook Post ===")
print(result["facebook_post"])
print("\n=== Instagram Post ===")
print(result["instagram_post"])
print("\n=== Twitter Post ===")
print(result["twitter_post"])
