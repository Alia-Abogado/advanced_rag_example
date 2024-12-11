from pprint import pprint
from dotenv import load_dotenv

load_dotenv()


from graph.chains.retrieval_grader import retrieval_grader, RetrievalGrader
from graph.chains.generation import generation_chain
from ingestion import retriever
from graph.chains.hallucination_grader import hallucination_grader, HallucinationGrader
from graph.chains.router import query_router, RouteQuery


def test_retrieval_grader_answer_yes() -> None:
    question = "hallucination"
    documents = retriever.invoke(question)
    document_text = documents[0].page_content
    res: RetrievalGrader = retrieval_grader.invoke(
        input={"question": question, "document": document_text}
    )  # type: ignore
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "hallucination"
    documents = retriever.invoke(question)
    document_text = documents[0].page_content
    res: RetrievalGrader = retrieval_grader.invoke(
        input={"question": "how to make a sandwich", "document": document_text}
    )  # type: ignore
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "hallucination"
    docs = retriever.invoke(question)
    res = generation_chain.invoke({"question": question, "context": docs})
    pprint(res)


def test_hallucination_grader_answer_yes() -> None:
    question = "hallucination"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    res: HallucinationGrader = hallucination_grader.invoke(
        {"question": question, "documents": docs, "generation": generation}
    )  # type: ignore
    assert res.binary_score == True


def test_hallucination_grader_answer_no() -> None:
    question = "hallucination"
    docs = retriever.invoke(question)
    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "question": question,
            "documents": docs,
            "generation": "The moon is made of green cheese.",
        }
    )  # type: ignore
    assert res.binary_score == False


def test_query_router_to_vectorstore() -> None:
    question = "hallucination in LLMs"
    res: RouteQuery = query_router.invoke({"question": question})  # type: ignore
    assert res.datasource == "vectorstore"


def test_query_router_to_websearch() -> None:
    question = "what is the weather in Tokyo?"
    res: RouteQuery = query_router.invoke({"question": question})  # type: ignore
    assert res.datasource == "websearch"
