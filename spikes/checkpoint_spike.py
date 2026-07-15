from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"


class State(TypedDict):
    steps: list[str]


def node_one(state: State) -> State:
    print(">>> running node_one")
    return {"steps": state["steps"] + ["node_one ran"]}


def node_two(state: State) -> State:
    print(">>> running node_two")
    return {"steps": state["steps"] + ["node_two ran"]}


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("node_one", node_one)
    builder.add_node("node_two", node_two)
    builder.add_edge(START, "node_one")
    builder.add_edge("node_one", "node_two")
    builder.add_edge("node_two", END)
    # interrupt_before pauses execution right before node_two runs,
    # so we can simulate "the process died here" and later resume it.
    return builder.compile(checkpointer=checkpointer, interrupt_before=["node_two"])


def main():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # creates the checkpoint tables if they don't exist yet
        graph = build_graph(checkpointer)

        thread_id = "spike-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        existing = graph.get_state(config)
        if existing.values:
            print("Found an existing checkpoint:", existing.values)
            print("Resuming...")
            result = graph.invoke(None, config)
        else:
            print("No checkpoint found. Starting a fresh run...")
            result = graph.invoke({"steps": []}, config)

        print("Final state after this run:", result)


if __name__ == "__main__":
    main()