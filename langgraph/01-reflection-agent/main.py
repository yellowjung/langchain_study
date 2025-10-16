from typing import List, Sequence, TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generation_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)



def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map = {END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == '__main__':
    print("Hello World")

