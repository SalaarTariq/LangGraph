from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


class State(TypedDict):
    name: str
    message: str


def chat_node(state: State) -> State:
    prompt = f"Hey, my name is {state['name']}. {state['message']}"
    response = llm.invoke(prompt)
    return {"name": state["name"], "message": response.content}


result = chat_node({"name": "Salaar", "message": "What is LangGraph in one sentence?"})
print(f"Name: {result['name']}")
print(f"Response: {result['message']}")
