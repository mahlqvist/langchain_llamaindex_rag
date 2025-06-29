import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import get_response_synthesizer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from dotenv import load_dotenv
import os


logging.basicConfig(level=logging.ERROR, filename="test.log", filemode="a",
                    format="%(asctime)s from %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

env_path = os.path.join(os.getcwd(), "config", ".env")

_ = load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

llm_heavy = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    top_p=0.9,
)

light_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
)


env_path = os.path.join(os.getcwd(), "config", ".env")
doc_path = os.path.join(os.getcwd(), "data", "dissociating_ai_from_ac.pdf")

documents = SimpleDirectoryReader(
	input_files=[doc_path]
).load_data()

node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

nodes = node_parser.get_nodes_from_documents(
	documents=documents,
	show_progress=False
)

db = chromadb.PersistentClient(path="chroma_db_test")

my_collection = db.get_or_create_collection(name="artificial_conscioussness")

vector_store = ChromaVectorStore(chroma_collection=my_collection)

storage_context = StorageContext.from_defaults(
	vector_store=vector_store
)

index = VectorStoreIndex(
	nodes=nodes,
	embed_model=embeddings,
	storage_context=storage_context
)

retriever = index.as_retriever(similarity_top_k=2)
retrieved_nodes = retriever.retrieve("Can machines think?")

print(retrieved_nodes)