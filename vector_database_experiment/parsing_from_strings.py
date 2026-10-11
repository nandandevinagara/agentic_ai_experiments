from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("source_files")

documents = [
    "Python developer working with PyTorch and LLMs",
    "Backend engineer working with FastAPI and PostgreSQL",
    "Robotics engineer working with ROS and computer vision",
    "nurse working with patient care and medical records",
    "political scientist working with data analysis and public policy"
]

embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=["file1", "file2", "file3", "file4", "file5"]
)

query = "I want to work in medical field"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(results["documents"])