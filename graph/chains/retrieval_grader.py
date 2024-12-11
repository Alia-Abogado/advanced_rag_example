from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class RetrievalGrader(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'",
    )


structured_llm_grader = llm.with_structured_output(RetrievalGrader)

system = """
You are a helpful assistant that scores the relevance of retrieved documents to a question.
If the document contains keywords or phrases that are relevant to the question, you should return 'yes'.
Otherwise, you should return 'no'.
Give a binary score 'yes' or 'no' to indicate if the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
