from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore"
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing a user question to a vectorstore or websearch.
The vectorstore contains documents about:
-the Mexican Customs Law (Ley Aduanera) which regulates customs operations, international trade, and the entry and exit of goods in Mexico. Key aspects covered include:
Customs procedures and requirements for importing/exporting goods
Roles and responsibilities of customs agents and customs agencies
Requirements for storing and handling goods in customs facilities
Documentation requirements for international trade
Rules for transportation companies and carriers
Handling of postal shipments
Customs duties, fees and taxes
Penalties and infractions related to customs operations
Electronic notification and documentation procedures
Special handling requirements for hazardous materials
Requirements for customs warehouses and storage facilities

The vectorstore could answer questions like:

What are the requirements for customs agents in Mexico?
How are hazardous materials handled in Mexican customs?
What documentation is needed for importing goods to Mexico?
What are the responsibilities of transportation companies under Mexican customs law?
How are postal shipments handled by Mexican customs?
What penalties exist for customs violations in Mexico?
What are the rules for customs warehouses in Mexico?

Your task is to determine whether a user's question should be routed to:
1. The vectorstore - if the question is likely answerable from our existing documents
2. Web search - for all other questions
"""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

query_router = route_prompt | structured_llm_router
