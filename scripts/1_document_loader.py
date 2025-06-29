from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

my_test_doc = Document(text="Hello, world!")
print(my_test_doc)

# Load pdf files and parse into LlamaIndex Document objects
documents = SimpleDirectoryReader(
	input_files=["./data/dissociating_ai_from_ac.pdf"]
).load_data()

# Splits text into sentences then merge into chunks
node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

# Splits chunks into nodes (a smaller semantic unit for indexing & retrieval)
nodes = node_parser.get_nodes_from_documents(
	documents=documents,
	show_progress=False
)

for i in range(5):
	print(nodes[i])