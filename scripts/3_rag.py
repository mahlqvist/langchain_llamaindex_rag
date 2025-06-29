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
doc_path = os.path.join(os.getcwd(), "data", "dissociating_ai_from_ac.pdf")

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

# Initialize ChromaDB client and collection
db = chromadb.PersistentClient(path="chroma_db_test")
my_collection = db.get_or_create_collection(name="artificial_consciousness")
vector_store = ChromaVectorStore(chroma_collection=my_collection)


# Check if collection is empty (first run)
if my_collection.count() == 0:
    documents = SimpleDirectoryReader(
        input_files=[doc_path]
    ).load_data()

    node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

    nodes = node_parser.get_nodes_from_documents(
        documents=documents,
        show_progress=False
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embeddings,
        storage_context=storage_context
    )
else:

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embeddings
    )


def retrieve_chunks(query: str):
    retriever = index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(query)

    return retrieved_nodes


extract_template = """You are a precision context extraction assistant. 
Your task is to:

1. Analyze the USER QUERY to:
   - Understand the current information need
2. Review all provided MEMORY CHUNKS:
   - Spot conceptual connections (even if different keywords are used)
3. Select only the most relevant information that:
   - Directly answers the USER QUERY
   - Maintains strong conversation continuity

Output requirements:
- First extract relevant context
- Never include information that isn't relevant
- If nothing is relevant, return "NO RELEVANT CONTEXT FOUND"

Processing steps you MUST follow:
1. First restate the USER QUERY in your own words to confirm understanding
2. Analyze each memory piece for relevance
3. Finally synthesize the CONTEXT from relevant fragments

Example output format:
<Understanding>[Your interpretation of the query]</Understanding>
<Context>
[Your extracted context with connection markers]
</Context>

Here is the USER QUERY:
{usr_query}

Here are the CHUNKS from the retrieved data:
{chunks}"""

retrieve_prompt = PromptTemplate.from_template(extract_template)

retrieve_chain = (
    retrieve_prompt
    | light_llm
    | StrOutputParser()
)


generation_template = """Generate a knowledgeable response to the query using context if available:

Context: {context}"""

generation_prompt = PromptTemplate.from_template(generation_template)

generation_chain = (
    generation_prompt
    | llm_heavy
    | StrOutputParser()
)


overall_chain = (
    # Start with original query
    RunnablePassthrough.assign(
        # Stage 1: Retrieve 
        chunks=lambda x: retrieve_chunks(x["usr_query"]),
        # Stage 2: Extract information
        extracted=lambda x: retrieve_chain.invoke(
            {"usr_query": x["usr_query"], 
             "chunks": retrieve_chunks(x["usr_query"])
            }
        )
    )
    # Stage 3: Generate final response
    | RunnableLambda(lambda x: generation_chain.invoke({
        "context": x["extracted"] 
    }))
)

# Test run
response = overall_chain.invoke({"usr_query": "Can machines think?"})

print(response)
