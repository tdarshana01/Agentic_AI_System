from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import json

# Load data
with open("past_incidents.json", "r") as f:
    data = json.load(f)

documents = []
for incident in data:
    text = (
        f"Date: {incident['date']}\n"
        f"Line: {incident['line']}\n"
        f"Description: {incident['description']}\n"
        f"Cause: {incident['cause']}\n"
        f"Remedy: {incident['remedy']}\n"
    )
    documents.append(Document(text=text, metadata={"incident_id": incident['incident_id']}))

print(f"Prepared {len(documents)} documents for indexing.")

# Embedding
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Build index and automatically create storage context
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Save everything
index.storage_context.persist(persist_dir="storage")

print("Saved FAISS index to faiss.index")

