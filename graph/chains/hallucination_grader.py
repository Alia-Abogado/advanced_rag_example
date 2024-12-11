from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class HallucinationGrader(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(HallucinationGrader)

system = """
You are a helpful assistant that scores the hallucination present in a generation answer.
You will receive:
1. A reference context or source document containing verified facts
2. A generated answer to evaluate

Your task is to determine if the generated answer contains any hallucinations - information that is not supported by or contradicts the reference context.

Evaluation Rules:
1. Compare ONLY against the provided reference context
2. Mark as hallucination (binary_score = False) if the answer:
   - States facts not present in the reference
   - Makes claims that contradict the reference
   - Invents specific details, numbers, dates, or quotes
   - Introduces new concepts or relationships not established in the reference

3. Do NOT mark as hallucination (binary_score = True) if the answer:
   - Rephrases or summarizes information from the reference
   - Makes general statements that logically follow from the reference
   - Uses different words to express the same concepts
   - Omits information (incompleteness is not hallucination)

4. When evaluating:
   - Focus on factual claims, not writing style or format
   - Consider both explicit statements and implied claims
   - Check each claim individually against the reference
   - Apply strict evaluation - if in doubt, mark as hallucination

Output:
You will output a structured response with binary_score:
- True if the answer is fully grounded in the reference context
- False if ANY hallucination is detected

Remember: Your goal is to ensure high-quality, factual responses by identifying any instances where the generated text makes claims beyond what is supported by the reference context."""


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
