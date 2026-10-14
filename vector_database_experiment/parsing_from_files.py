from urllib import response

from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import ollama

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("source_files")


def get_source_documents():
    documents = []

    for file in Path("vector_database_experiment/source_files").glob("*.txt"):
        text = file.read_text()
        documents.append(text)
    return documents

def find_closest_documents(sentence_embedding_model, query, n_results=2):
    query_embedding = sentence_embedding_model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results["documents"][0]  # Return the top most result #type: ignore


def rag_by_ollama(prompt, model_name="qwen3:4b"):
    # Send the prompt to the LLM
    response = ollama.chat(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )

    print("Answer:", response["message"]["content"])


def rag_by_transformers(prompt, model_name="Qwen/Qwen3-4B"):
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, device_map="cpu"
    )

    # prompt = "Explain vector databases in simple terms."

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100)  # type : ignore

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def get_list_dimensions(values):
    dimensions = []
    while isinstance(values, list):
        dimensions.append(len(values))
        if not values:
            break
        values = values[0]
    return dimensions


embeddings = sentence_embedding_model.encode(get_source_documents()).tolist()

collection.add(
    documents=get_source_documents(),
    embeddings=embeddings,
    ids=[f"file{i}" for i in range(len(get_source_documents()))],
)

# query = "whats is the size of cache folder of hugging face in total?"
query = "Could you tell if the capital of France is mentioned in your source of files?"

# Extract the retrieved documents
retrieved_documents = find_closest_documents(sentence_embedding_model, query, n_results=2)

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

way = "transformers"  # or "ollama"
if way == "ollama":
    rag_by_ollama(prompt)
else:
    rag_by_transformers(prompt)