from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=10)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("Running web search...")
    question = state["question"]
    documents = state["documents"]
    results = web_search_tool.invoke(question)
    joined_results = "\n".join([result["content"] for result in results])
    web_results = Document(page_content=joined_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
