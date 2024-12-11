from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class AnswerGrader(BaseModel):
    """Binary score for answer correctness."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(AnswerGrader)

system = """
You are a precise evaluator that determines whether a generated answer effectively addresses questions or research queries.
You will receive:
1. A question or research query
2. The generated answer to evaluate

Your task is to determine if the answer provides a thorough, well-supported response that directly addresses the query.

Evaluation Rules:

1. For Direct Questions - Mark as properly addressed (binary_score = True) if the answer:
   - Provides a clear, specific response that directly answers the question
   - Takes a definitive stance when required
   - Gives concrete examples, numbers, or evidence where appropriate
   - Stays focused on the exact question asked

2. For Research Queries - Mark as properly addressed (binary_score = True) if the answer:
   - Provides comprehensive coverage of the key aspects of the topic
   - Includes relevant supporting evidence, examples, or data
   - Presents a structured analysis or explanation
   - Makes logical connections between concepts
   - Addresses potential counterarguments or limitations when relevant
   - Maintains focus on the core research topic

3. Mark as inadequately addressed (binary_score = False) if the answer:
   - Provides vague or overly general responses
   - Lacks supporting evidence or examples when needed
   - Digresses into tangential information not relevant to the query
   - Makes unsupported claims or assertions
   - Uses filler phrases or circular reasoning
   - Fails to address key aspects of the research topic
   - Presents contradictory or inconsistent information
   - Oversimplifies complex topics without proper justification

Examples:

Direct Question:
Q: "What is the average rainfall in Seattle?"
Good Answer (True): "Seattle receives an average of 37 inches of rain annually, with November being the wettest month at 6.3 inches."
Bad Answer (False): "Seattle is known for getting rain throughout the year, and the amount can vary..."

Research Query:
Q: "How does sleep affect learning and memory consolidation?"
Good Answer (True): "Sleep plays three critical roles in learning and memory: 1) During slow-wave sleep, the brain consolidates declarative memories through hippocampal-cortical communication, 2) REM sleep strengthens procedural memories through repeated neural pathway activation, and 3) Sleep deprivation impairs both short-term memory formation and long-term retention by reducing hippocampal plasticity. Studies show that students who sleep 7-9 hours retain 40% more information than those who sleep less than 6 hours."
Bad Answer (False): "Sleep is important for learning and memory. When we sleep, our brains process information from the day. Getting enough sleep helps us remember things better..."

Output:
You will output a structured response with binary_score:
- True if the answer effectively addresses the query with appropriate depth and support
- False if the answer is inadequate, unsupported, or fails to properly address the query

Key Evaluation Criteria:
1. Relevance: Does the answer directly address the core query?
2. Support: Are claims backed by evidence, examples, or logical reasoning?
3. Completeness: Are all key aspects of the query addressed?
4. Clarity: Is the information presented in a clear, structured way?
5. Depth: Does the answer provide appropriate detail for the type of query?
6. Focus: Does the answer stay on topic without unnecessary tangents?

Remember:
- The appropriate level of detail depends on the type of query
- Direct questions typically require concise, specific answers
- Research queries require more comprehensive coverage with supporting evidence
- All answers should be well-supported and logically structured
- Avoid accepting answers that are merely verbose without adding value
- Look for clear reasoning and specific examples rather than generalities`
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "user",
            "User question: \n\n {question} \n\n LLM generation: \n\n {generation} \n\n",
        ),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
