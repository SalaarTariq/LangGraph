from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    messages: Annotated[list, add_messages]


generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that creates social media posts."),
    ("human", "Write a post about {topic}."),
])

curate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that improves social media posts."),
    ("human", "Curate and improve this post, make it more engaging:\n\n{post}"),
])


generate_chain = generate_prompt | llm
curate_chain = curate_prompt | llm


def generate_post(state: State) -> State:
    response = generate_chain.invoke({"topic": state["messages"][-1].content})
    return {"messages": [response]}


def curate_post(state: State) -> State:
    response = curate_chain.invoke({"post": state["messages"][-1].content})
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("generate_post", generate_post)
graph.add_node("curate_post", curate_post)
graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "curate_post")
graph.add_edge("curate_post", END)

app = graph.compile()

result = app.invoke({"messages": [("user", "data privacy")]})

for msg in result["messages"]:
    print(f"\n--- {msg.type} ---")
    print(msg.content)