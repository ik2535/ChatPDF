import numpy as np
from langchain.embeddings import AmazonTitanEmbeddings
from langchain.llms import Anthropic
import faiss

# Placeholder: load original text mapping
stored_texts = []

index = faiss.read_index('faiss.index')
embedder = AmazonTitanEmbeddings(model_name='titan-embedding-text-v1')

def find_contextual_snippets(query: str, k: int = 5) -> str:
    qvec = embedder.embed_query(query)
    D, I = index.search(np.array([qvec]).astype('float32'), k)
    snippets = [stored_texts[i] for i in I[0]]
    return '\n'.join(snippets)

def ask_llm(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    llm = Anthropic(model="claude-2.1")
    return llm(prompt)
