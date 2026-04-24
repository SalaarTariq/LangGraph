from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    topic: str
    platform: str
    facebook_post: str
    instagram_post: str
    twitter_post: str


def router(state: State) -> State:
    response = llm.invoke(
        f"User request: '{state['topic']}'\n\n"
        "Your job: check if the user specifically mentioned one of these platforms: facebook, instagram, twitter.\n"
        "Examples:\n"
        "- 'write a tweet about AI' -> twitter\n"
        "- 'make an instagram post about food' -> instagram\n"
        "- 'create a facebook post about tech' -> facebook\n"
        "- 'write a post about AI' -> all\n"
        "- 'create content about data privacy' -> all\n\n"
        "Reply with ONLY one word: facebook, instagram, twitter, or all."
    )
    platform = response.content.strip().lower()
    if platform not in ("facebook", "instagram", "twitter", "all"):
        platform = "all"
    return {"platform": platform}


def route_to_platform(state: State) -> list[str]:
    if state["platform"] == "all":
        return ["facebook", "instagram", "twitter"]
    return [state["platform"]]


def facebook_node(state: State) -> State:
    response = llm.invoke(f"Write a Facebook post about: {state['topic']}. Keep it casual and engaging.")
    return {"facebook_post": response.content}


def instagram_node(state: State) -> State:
    response = llm.invoke(f"Write an Instagram caption about: {state['topic']}. Use emojis and hashtags.")
    return {"instagram_post": response.content}


def twitter_node(state: State) -> State:
    response = llm.invoke(f"Write a tweet about: {state['topic']}. Keep it under 280 characters.")
    return {"twitter_post": response.content}


# Graph:
# START -> router -> "all"       -> facebook + instagram + twitter (parallel) -> END
#                 -> "facebook"  -> facebook  -> END
#                 -> "instagram" -> instagram -> END
#                 -> "twitter"   -> twitter   -> END

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

app = graph.compile()

mermaid = app.get_graph().draw_mermaid()
print(mermaid)

img_bytes = app.get_graph().draw_mermaid_png()
with open("codes/graph_routing.png", "wb") as f:
    f.write(img_bytes)
print("Graph saved to codes/graph_routing.png")

topic = input("\nEnter your topic: ")

result = app.invoke({"topic": topic, "platform": "", "facebook_post": "", "instagram_post": "", "twitter_post": ""})

print(f"\nPlatform routed: {result['platform']}")
if result["facebook_post"]:
    print(f"\n=== Facebook Post ===\n{result['facebook_post']}")
if result["instagram_post"]:
    print(f"\n=== Instagram Post ===\n{result['instagram_post']}")
if result["twitter_post"]:
    print(f"\n=== Twitter Post ===\n{result['twitter_post']}")
