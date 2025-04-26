from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AmazonTitanEmbeddings
import faiss
import numpy as np

def chunk_and_normalize(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

class FAISSIndex:
    def __init__(self, embeddings, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.embeddings = embeddings

    def add(self, texts: List[str]):
        vecs = [self.embeddings.embed_query(t) for t in texts]
        self.index.add(np.array(vecs).astype('float32'))

    def save_local(self, path: str):
        faiss.write_index(self.index, path)

def create_local_vector_index(chunks: List[str]) -> FAISSIndex:
    embedder = AmazonTitanEmbeddings(model_name='titan-embedding-text-v1')
    dim = embedder.embed_query('test').shape[0]
    idx = FAISSIndex(embedder, dim)
    idx.add(chunks)
    return idx
