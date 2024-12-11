from typing import Any, Dict

from graph.chains.retrieval_grader import RetrievalGrader, retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Grade retrieved documents for relevance to the question.
    If the document is not relevant, it will set a flag in the state to run a web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """

    print("Checking relevance of retrieved documents...")
    question = state["question"]
    documents = state["documents"]
    web_search = False
    filtered_documents = []
    if documents is not None:
        for document in documents:
            score: RetrievalGrader = retrieval_grader.invoke(
                input={"question": question, "document": document}
            )  # type: ignore
            grade = score.binary_score.lower()
            if grade == "no":
                web_search = True
            else:
                filtered_documents.append(document)
    else:
        web_search = True
    return {
        "documents": filtered_documents,
        "question": question,
        "web_search": web_search,
    }
