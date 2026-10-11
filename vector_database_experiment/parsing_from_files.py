from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import ollama

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("source_files")

documents = []

for file in Path("vector_database_experiment/source_files").glob("*.txt"):
    text = file.read_text()
    documents.append(text)

embeddings = model.encode(documents).tolist()

def get_list_dimensions(values):
    dimensions = []
    while isinstance(values, list):
        dimensions.append(len(values))
        if not values:
            break
        values = values[0]
    return dimensions
# print('embeddings generated for documents ', embeddings)
print('dimensions of embeddings: ', get_list_dimensions(embeddings))

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"file{i}" for i in range(len(documents))]
)

query = "whats is the size of cache folder of hugging face in total?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("closest documents:", results["documents"])

# Extract the retrieved documents
retrieved_documents = results["documents"][0]   # top most result

# Combine the documents into context
context = "\n\n".join(retrieved_documents)

# Construct the prompt
prompt = f"""
Answer the question using only the context provided below.

If the answer is not present in the context, say that you don't
have enough information to answer.

Context:
{context}

Question:
{query}
"""

# Send the prompt to the LLM
response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("Answer:", response["message"]["content"])