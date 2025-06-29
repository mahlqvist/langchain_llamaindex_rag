from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

my_test_doc = Document(text="Hello, world!")


# Load pdf files
documents = SimpleDirectoryReader(
	input_files=["./data/dissociating_ai_from_ac.pdf"]
).load_data()

# 
node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

nodes = node_parser.get_nodes_from_documents(
	documents=documents,
	show_progress=False
)

for i in range(5):
	print(nodes[i])