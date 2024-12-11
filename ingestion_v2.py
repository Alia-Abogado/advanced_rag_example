import os
import re
import bisect
from typing import List, Dict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate


from langchain_community.document_loaders import PyPDFLoader
from collections import defaultdict

############################################
# Load PDF and Extract Structured Chunks
############################################

# loader = PyPDFLoader("ley_aduanera.pdf")
# pages = loader.load()

# page_texts = [p.page_content for p in pages]

# full_text = ""
# page_start_offsets = []
# cumulative_len = 0

# title_changes = []
# chapter_changes = []

# current_titulo_num = "UNKNOWN_TITULO_NUM"
# current_titulo_name = "UNKNOWN_TITULO_NAME"
# current_capitulo_num = "UNKNOWN_CAPITULO_NUM"
# current_capitulo_name = "UNKNOWN_CAPITULO_NAME"

# for p_text in page_texts:
#     page_start_offsets.append(cumulative_len)
#     lines = p_text.split("\n")
#     for i, line in enumerate(lines):
#         line_stripped = line.strip()

#         # Detect Título
#         titulo_match = re.match(r"(?i)^Título\s+([\wÁÉÍÓÚÜáéíóúü]+)", line_stripped)
#         if titulo_match:
#             current_titulo_num = titulo_match.group(1)
#             titulo_name = "UNKNOWN_TITULO_NAME"
#             # Peek next line for Título name
#             if i + 1 < len(lines):
#                 next_line = lines[i + 1].strip()
#                 if next_line:
#                     titulo_name = next_line
#             current_titulo_name = titulo_name
#             title_changes.append(
#                 (cumulative_len, current_titulo_num, current_titulo_name)
#             )

#         # Detect Capítulo
#         capitulo_match = re.match(r"(?i)^Capítulo\s+([\wÁÉÍÓÚÜáéíóúü]+)", line_stripped)
#         if capitulo_match:
#             current_capitulo_num = capitulo_match.group(1)
#             capitulo_name = "UNKNOWN_CAPITULO_NAME"
#             if i + 1 < len(lines):
#                 next_line = lines[i + 1].strip()
#                 if next_line:
#                     capitulo_name = next_line
#             current_capitulo_name = capitulo_name
#             chapter_changes.append(
#                 (cumulative_len, current_capitulo_num, current_capitulo_name)
#             )

#         full_text += line + "\n"
#         cumulative_len += len(line) + 1

# # Split by Artículos
# articles = re.split(r"(?i)(?=ARTICULO\s*\d+o?\.?-?[A-Za-z]?)", full_text)
# articles = [a.strip() for a in articles if a.strip()]

# used_positions = []


# def find_offset(substring):
#     start = 0
#     while True:
#         pos = full_text.find(substring[:30], start)
#         if pos == -1:
#             return -1
#         if full_text[pos : pos + len(substring)] == substring:
#             if all(not (pos <= up < pos + len(substring)) for up in used_positions):
#                 used_positions.extend(range(pos, pos + len(substring)))
#                 return pos
#         start = pos + 1


# def get_current_title_info(offset):
#     idx = bisect.bisect_right(title_changes, (offset,)) - 1
#     if idx >= 0:
#         return title_changes[idx][1], title_changes[idx][2]
#     else:
#         return "UNKNOWN_TITULO_NUM", "UNKNOWN_TITULO_NAME"


# def get_current_chapter_info(offset):
#     idx = bisect.bisect_right(chapter_changes, (offset,)) - 1
#     if idx >= 0:
#         return chapter_changes[idx][1], chapter_changes[idx][2]
#     else:
#         return "UNKNOWN_CAPITULO_NUM", "UNKNOWN_CAPITULO_NAME"


# def get_page_number(offset):
#     page_index = bisect.bisect_right(page_start_offsets, offset) - 1
#     if page_index < 0:
#         page_index = 0
#     return page_index + 1


# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# docs: List[Document] = []
# for article_text in articles:
#     article_offset = find_offset(article_text)
#     if article_offset == -1:
#         article_offset = 0
#     article_match = re.search(r"(?i)ARTICULO\s*([\d]+o?\.?-?[A-Za-z]?)", article_text)
#     article_number = article_match.group(1) if article_match else "UNKNOWN"

#     titulo_num, titulo_name = get_current_title_info(article_offset)
#     capitulo_num, capitulo_name = get_current_chapter_info(article_offset)

#     chunks = splitter.split_text(article_text)
#     local_offset = 0
#     for i, chunk in enumerate(chunks):
#         chunk_start = article_offset + local_offset
#         page_number = get_page_number(chunk_start)
#         docs.append(
#             Document(
#                 page_content=chunk,
#                 metadata={
#                     "article_number": article_number,
#                     "titulo_number": titulo_num,
#                     "titulo_name": titulo_name,
#                     "capitulo_number": capitulo_num,
#                     "capitulo_name": capitulo_name,
#                     "chunk_index": i,
#                     "page_number": page_number,
#                     "source": "ley_aduanera.pdf",
#                 },
#             )
#         )
#         local_offset += len(chunk)


# print("Number of chunks created:", len(docs))


############################################
# LLM Setup
############################################

# llm = ChatOpenAI(model="gpt-4o-mini")  # Replace with a local model if needed
embeddings = HuggingFaceEmbeddings(
    model_name="law-ai/InCaseLawBERT", cache_folder="./hf_cache"
)

############################################
# Hierarchical Summarization
############################################
# We'll summarize at Article -> Capítulo -> Título -> Global

# Group chunks by article_number
# articles_dict = defaultdict(list)
# for d in docs:
#     articles_dict[d.metadata["article_number"]].append(d)


# # Summarize a list of documents into a single summary
# def summarize_texts(texts: List[str], prompt_template: str) -> str:
#     summarize_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 prompt_template,
#             )
#         ]
#     )
#     summarize_chain = summarize_prompt | llm | StrOutputParser()
#     batch_text = "\n\n".join(texts)
#     return summarize_chain.invoke(input={"text": batch_text}).strip()


# article_summaries = {}
# article_prompt = """
# Please summarize the following article text into a concise summary capturing its main points:
# {text}
# """

# for art_num, art_docs in articles_dict.items():
#     art_texts = [d.page_content for d in art_docs]
#     article_summaries[art_num] = summarize_texts(art_texts, article_prompt)

# print("Article summaries:")
# for art_num, summary in article_summaries.items():
#     print(f"Article {art_num}: {summary}")


# # Group articles by capítulo
# capitulos_dict = defaultdict(list)
# for art_num, summary in article_summaries.items():
#     # Find a doc for this article to get its capítulo
#     example_doc = articles_dict[art_num][0]
#     cap_key = (
#         example_doc.metadata["titulo_number"],
#         example_doc.metadata["capitulo_number"],
#     )
#     capitulos_dict[cap_key].append((art_num, summary))

# capitulo_summaries = {}
# capitulo_prompt = """
# Summarize the following article summaries into a concise capítulo-level summary:
# {text}
# """

# for cap_key, art_list in capitulos_dict.items():
#     cap_texts = [f"Artículo {a[0]}: {a[1]}" for a in art_list]
#     capitulo_summaries[cap_key] = summarize_texts(cap_texts, capitulo_prompt)


# # Group capítulos by título
# titulos_dict = defaultdict(list)
# for cap_key, cap_sum in capitulo_summaries.items():
#     titulo_num = cap_key[0]
#     titulos_dict[titulo_num].append(cap_sum)

# titulo_summaries = {}
# titulo_prompt = """
# Summarize the following capítulo summaries into a concise título-level summary:
# {text}
# """

# for t_num, cap_sums in titulos_dict.items():
#     titulo_summaries[t_num] = summarize_texts(cap_sums, titulo_prompt)


# # Finally, global summary
# global_prompt = """
# Summarize the following título-level summaries into a global document summary:
# {text}
# """

# global_summary = summarize_texts(list(titulo_summaries.values()), global_prompt)

############################################
# Contextual Retrieval Prompt for Each Chunk
############################################
# For each chunk, we will generate a contextual snippet using:
# - global_summary
# - relevant título summary
# - relevant capítulo summary

# context_prompt = PromptTemplate(
#     input_variables=["global_summary", "titulo_summary", "capitulo_summary", "chunk"],
#     template="""
# Given the following hierarchical summaries:

# Global Summary:
# {global_summary}

# Título-level Summary:
# {titulo_summary}

# Capítulo-level Summary:
# {capitulo_summary}

# Here is a chunk of the document:
# {chunk}

# Explain succinctly what this chunk represents in the context of the full document, referencing the hierarchical structure above.
# """,
# )

# context_chain = context_prompt | llm | StrOutputParser()


# # Precompute título and capítulo summaries in a lookup
# def get_titulo_summary(t_num):
#     return titulo_summaries.get(t_num, "No título summary available.")


# def get_capitulo_summary(t_num, c_num):
#     return capitulo_summaries.get((t_num, c_num), "No capítulo summary available.")


# contextualized_chunks = []
# for d in docs:
#     t_num = d.metadata["titulo_number"]
#     c_num = d.metadata["capitulo_number"]

#     chunk_context = context_chain.invoke(
#         input={
#             "global_summary": global_summary,
#             "titulo_summary": get_titulo_summary(t_num),
#             "capitulo_summary": get_capitulo_summary(t_num, c_num),
#             "chunk": d.page_content,
#         }
#     )
#     full_context_chunk = chunk_context.strip() + " " + d.page_content
#     # Create a new Document with extended context
#     contextualized_chunks.append(
#         Document(page_content=full_context_chunk, metadata=d.metadata)
#     )

############################################
# Indexing and Retrieval
############################################
# Create vector store with contextualized chunks
# vectorstore = Chroma.from_documents(
#     docs,
#     embeddings,
#     collection_name="ley_aduanera",
#     persist_directory="./.chroma",
# )

vector_retriever = Chroma(
    collection_name="ley_aduanera",
    embedding_function=embeddings,
    persist_directory="./.chroma",
).as_retriever()

# # Create BM25 index
# bm25_retriever = BM25Retriever.from_documents(docs)

# # Example query
# query = "Quiero redactar una demanda en contra de una resolución que determine un crédito fiscal al término de una revisión de gabinete, ¿Que fundamentos legales pueden utilizarse para realizar la demanda?"

# # Vector retrieval
# vector_results = vector_retriever.invoke(query, k=10)

# print("Vector results:")
# for r in vector_results:
#     print(r.metadata["page_number"])

# # BM25 retrieval
# bm25_results = bm25_retriever.invoke(query)

# print("BM25 results:")
# for r in bm25_results:
#     print(r.metadata["page_number"])

# # Reciprocal Rank Fusion
# from collections import defaultdict


# def reciprocal_rank_fusion(*list_of_list_ranks_system, K=60):
#     rrf_map = defaultdict(float)
#     for rank_list in list_of_list_ranks_system:
#         for rank, doc in enumerate(rank_list, 1):
#             doc_id = doc.page_content
#             rrf_map[doc_id] += 1 / (rank + K)
#     sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
#     # Reconstruct docs from content
#     doc_map = {}
#     for r_list in list_of_list_ranks_system:
#         for d in r_list:
#             doc_map[d.page_content] = d
#     return [doc_map[s[0]] for s in sorted_items]


# hybrid_docs = reciprocal_rank_fusion(vector_results, bm25_results)
# final_docs = hybrid_docs[:5]  # Optionally re-rank with a re-ranker model if available

# ############################################
# # Generate Final Answer
# ############################################


# answer_chain_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a knowledgeable legal expert. You're to answer the question below using ONLY the provided context.
#              For every piece of context you are given, you must use it to answer the question, citing the context in your answer (article, section, etc).
#             Follow the instructions below carefully:
#             1. Begin by silently thinking step-by-step about how to answer. (Do not output these thoughts directly.)
#             2. Generate useful internal knowledge from the Context. Summarize relevant statutes, procedural steps, and any key legal references from the Context that could answer the Question. (Do not output this knowledge section directly in your final answer.)
#             3. Based on the internal knowledge you've generated, produce a private directional hint or mental note to improve clarity, specificity, and completeness in the final answer. (Do not output this hint directly.)
#             4. After these internal reasoning steps, produce a well-structured, comprehensive, and lawyerly answer that directly addresses the Question. Your final answer must rely solely on information from the Context and must not include speculation. Reference specific legal articles, procedures, and timeframes if they are mentioned in the Context. Do not include the internal knowledge or the hint in your final output.
#             """,
#         ),
#         (
#             "user",
#             """
#             Context:
#             {context}

#             Question: {query}
#             Let's think step-by-step.
#             Answer:
#             """,
#         ),
#     ]
# )

# answer_chain = answer_chain_prompt | llm | StrOutputParser()
# combined_context = "\n\n".join([d.page_content for d in final_docs])
# answer = answer_chain.invoke(
#     input={"context": combined_context, "query": query}
# ).strip()

# print("Final Answer:\n", answer)
