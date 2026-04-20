import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Define the state structure for our graph
# add_messages is a "reducer" that tells the graph how to update the 'messages' list
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a ChatGroq instance using settings from .env
# This will look for GROQ_API_KEY by default
llm = ChatGroq(model="llama3-8b-8192")

def chatbot(state: State):
    """
    Our main node: it takes the current messages, 
    passes them to the Groq LLM, and returns the AI's response.
    """
    return {"messages": [llm.invoke(state["messages"])]}

# Build the Graph
workflow = StateGraph(State)

# Add our only node
workflow.add_node("chatbot", chatbot)

# Connect the nodes
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# Compile the graph into an executable app
app = workflow.compile()

def main():
    print("--- Starting LangGraph Basic Chat ---")
    
    # Initial input
    initial_input = {"messages": [("user", "Hello! Can you explain LangGraph in one sentence?")]}
    
    # Run the graph and stream the output
    # 'events' will contain updates from each node
    for event in app.stream(initial_input):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    main()
