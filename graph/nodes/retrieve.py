from typing import Any, Dict

from graph.state import GraphState

# from ingestion import retriever
from ingestion_v2 import vector_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieving---")
    question = state["question"]
    documents = vector_retriever.invoke(question)
    return {"documents": documents, "question": question}
