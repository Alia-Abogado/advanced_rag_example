from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader, HallucinationGrader
from graph.chains.answer_grader import answer_grader, AnswerGrader
from graph.chains.router import query_router, RouteQuery

load_dotenv()

graph = StateGraph(GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENTS, grade_documents)
graph.add_node(WEBSEARCH, web_search)
graph.add_node(GENERATE, generate)


def should_web_search(state: GraphState) -> str:
    should_web_search = state["web_search"]
    if should_web_search:
        return WEBSEARCH
    else:
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("Checking for hallucinations...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score: HallucinationGrader = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )  # type: ignore

    if score.binary_score:
        print("Decision: generation is not hallucinated")
        grade: AnswerGrader = answer_grader.invoke(
            {"question": question, "generation": generation}
        )  # type: ignore
        if grade.binary_score:
            print("Decision: generation is correct")
            return "useful"
        else:
            print("Decision: generation is incorrect")
            return "not useful"
    else:
        print("Decision: generation is hallucinated")
        return "not supported"


def route_question(state: GraphState) -> str:
    question = state["question"]
    source: RouteQuery = query_router.invoke({"question": question})  # type: ignore
    if source.datasource == "vectorstore":
        print("Routing to vectorstore")
        return RETRIEVE
    elif source.datasource == "websearch":
        print("Routing to websearch")
        return WEBSEARCH
    else:
        raise ValueError(f"Invalid datasource: {source.datasource}")


graph.set_conditional_entry_point(
    route_question, path_map={WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_conditional_edges(
    GRADE_DOCUMENTS,
    should_web_search,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
graph.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={"useful": END, "not useful": WEBSEARCH, "not supported": GENERATE},
)

graph.add_edge(WEBSEARCH, GENERATE)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
