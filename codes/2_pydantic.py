from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

class LinkedInPostState(BaseModel):
    topic: str
    post: str
    currated_post: str


def generate_post(state: LinkedInPostState) -> LinkedInPostState:
    response = llm.invoke(f"Write a LinkedIn post about: {state.topic}")
    return LinkedInPostState(topic=state.topic, post=response.content, currated_post=state.currated_post)


def curate_post(state: LinkedInPostState) -> LinkedInPostState:
    response = llm.invoke(f"Curate and improve this LinkedIn post, make it more engaging and professional:\n\n{state.post}")
    return LinkedInPostState(topic=state.topic, post=state.post, currated_post=response.content)


graph = StateGraph(LinkedInPostState)
graph.add_node("generate_post", generate_post)
graph.add_node("curate_post", curate_post)
graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "curate_post")
graph.add_edge("curate_post", END)

app = graph.compile()

print(app.get_graph().draw_mermaid())

result = app.invoke({"topic": "Importance of AI", "post": "", "currated_post": ""})
print(f"\nTopic: {result['topic']}")
print(f"\nGenerated Post:\n{result['post']}")
print(f"\nCurated Post:\n{result['currated_post']}")