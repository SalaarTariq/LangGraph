from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    messages_manual: list
    messages_auto: Annotated[list, add_messages]


def generate_post(state: State) -> State:
    response = llm.invoke(state["messages_auto"])
    return {
        "messages_manual": [response],
        "messages_auto": [response],
    }


def curate_post(state: State) -> State:
    curate_prompt = "Curate and improve this LinkedIn post, make it more engaging and professional."
    response = llm.invoke(state["messages_auto"] + [("user", curate_prompt)])
    return {
        "messages_manual": [response],
        "messages_auto": [response],
    }


graph = StateGraph(State)
graph.add_node("generate_post", generate_post)
graph.add_node("curate_post", curate_post)
graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "curate_post")
graph.add_edge("curate_post", END)

app = graph.compile()

result = app.invoke({
    "messages_manual": [("user", "Write a LinkedIn post about Importance of AI")],
    "messages_auto": [("user", "Write a LinkedIn post about Importance of AI")],
})

print("\n--- messages_manual ---")
print(result["messages_manual"])
print("\n--- messages_auto ---")
print(result["messages_auto"])
