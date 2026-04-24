import operator
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


# Main graph state
class State(TypedDict):
    query: str
    sub_tasks: list[str]
    results: Annotated[list, operator.add]
    final_response: str


# Worker receives its own smaller state
class WorkerState(TypedDict):
    sub_task: str


# Node 1: Orchestrator - breaks query into sub-tasks
def orchestrator(state: State) -> State:
    response = llm.invoke(
        f"Break this query into 2-4 smaller independent sub-tasks.\n"
        f"Query: {state['query']}\n\n"
        "Return ONLY the sub-tasks, one per line. No numbering, no bullets, no extra text."
    )
    sub_tasks = [t.strip() for t in response.content.strip().split("\n") if t.strip()]
    print(f"\nOrchestrator created {len(sub_tasks)} sub-tasks:")
    for i, task in enumerate(sub_tasks, 1):
        print(f"  {i}. {task}")
    return {"sub_tasks": sub_tasks}


# Conditional edge: dynamically send each sub-task to a worker
def assign_workers(state: State) -> list[Send]:
    return [Send("worker", {"sub_task": task}) for task in state["sub_tasks"]]


# Node 2: Worker - answers a single sub-task
def worker(state: WorkerState) -> dict:
    print(f"\nWorker processing: {state['sub_task'][:50]}...")
    response = llm.invoke(
        f"Answer this specific task in 2-3 sentences:\n{state['sub_task']}"
    )
    return {"results": [response.content]}


# Node 3: Collector - synthesizes all results into final response
def collector(state: State) -> State:
    all_results = "\n\n".join(state["results"])
    response = llm.invoke(
        f"Original query: {state['query']}\n\n"
        f"Here are the research results:\n{all_results}\n\n"
        "Combine these into one clear, well-structured response for the user."
    )
    return {"final_response": response.content}


# Build the graph
# START -> orchestrator -> [worker, worker, ...] (parallel via Send) -> collector -> END

graph = StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("collector", collector)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_workers, ["worker"])
graph.add_edge("worker", "collector")
graph.add_edge("collector", END)

app = graph.compile()

# Generate graph visualization
mermaid = app.get_graph().draw_mermaid()
print(mermaid)

try:
    img_bytes = app.get_graph().draw_mermaid_png(max_retries=3, retry_delay=2.0)
    with open("codes/graph_orchestrator.png", "wb") as f:
        f.write(img_bytes)
    print("Graph saved to codes/graph_orchestrator.png\n")
except Exception:
    print("Could not generate PNG (mermaid.ink timed out). Use the mermaid text above instead.\n")

# Run
query = input("Enter your query: ")

result = app.invoke({"query": query, "sub_tasks": [], "results": [], "final_response": ""})

print("\n=== Final Response ===")
print(result["final_response"])
