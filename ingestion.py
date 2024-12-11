from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    "https://www.falkordb.com/blog/advanced-rag/",
    "https://www.datacamp.com/blog/rag-advanced",
    "https://praveengovindaraj.com/thread-of-thought-thot-a-new-prompting-approach-to-complex-contexts-f6827e1aec1f",
    "https://mirascope.com/tutorials/prompt_engineering/text_based/thread_of_thought/",
]

pdfs = [
    "./thot.pdf",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=10,
)

chunks = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     collection_name="rag-collection",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-collection",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
