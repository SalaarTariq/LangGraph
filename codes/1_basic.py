from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    name: str
    message: str


def welcome_node(state: State) -> State:
    prompt = f"Hey, my name is {state['name']}. {state['message']}"
    response = llm.invoke(prompt)
    return {"name": state["name"], "message": response.content}


graph = StateGraph(State)
graph.add_node("welcome", welcome_node)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

app = graph.compile()

print(app.get_graph().draw_mermaid())

result = app.invoke({"name": "Salaar", "message": "What is LangGraph in one sentence?"})
print(f"Name: {result['name']}")
print(f"Response: {result['message']}")
