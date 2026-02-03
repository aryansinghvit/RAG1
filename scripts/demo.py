#!/usr/bin/env python3
"""scripts/demo.py
Run a small end-to-end demo: ingest PDFs -> split -> embed -> store -> retrieve -> (optional) LLM

Usage:
  python scripts/demo.py --data-dir data --persist-dir data/vector_store --model llama-3.1-8b-instant

Notes:
- Use a .env file (not committed) or set GROQ_API_KEY in the environment to enable LLM calls.
- Use --skip-llm to skip LLM calls (useful for CI without credentials).
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Minimal imports used by the demo
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    import chromadb
except Exception as e:
    print("Missing required packages. Run: pip install -r requirements.txt")
    raise

import uuid
import numpy as np

# Load .env if present
load_dotenv()

# -----------------------------
# Pipeline utilities (minimal)
# -----------------------------

def process_all_pdfs(pdf_directory: str):
    pdf_dir = Path(pdf_directory)
    files = list(pdf_dir.glob("**/*.pdf"))
    docs = []
    print(f"Found {len(files)} PDF files in {pdf_directory}")
    for p in files:
        print(f" - loading: {p}")
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        for pg in pages:
            pg.metadata.setdefault('source_file', p.name)
            pg.metadata.setdefault('file_type', 'pdf')
        docs.extend(pages)
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded")

    def generate_embeddings(self, texts):
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        emb = self.model.encode(texts, show_progress_bar=False)
        return np.array(emb)


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, docs, embeddings):
        ids, metadatas, documents_text, emb_list = [], [], [], []
        for i, (d, e) in enumerate(zip(docs, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            met = dict(d.metadata) if hasattr(d, 'metadata') else {}
            met['length'] = len(getattr(d, 'page_content', str(d))[:1000])
            metadatas.append(met)
            documents_text.append(getattr(d, 'page_content', str(d)))
            emb_list.append(e.tolist())
        self.collection.add(ids=ids, embeddings=emb_list, metadatas=metadatas, documents=documents_text)
        print(f"Added {len(ids)} documents to collection")

    def query(self, query_embedding, top_k=5):
        res = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        return res


class RAGRetriever:
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k=5):
        emb = self.embedding_manager.generate_embeddings([query])[0]
        res = self.vectorstore.query(emb, top_k=top_k)
        docs = []
        if res and res.get('documents') and res['documents'][0]:
            docs_raw = res['documents'][0]
            metas = res['metadatas'][0]
            distances = res['distances'][0]
            ids = res['ids'][0]
            for i, (doc, meta, dist, id_) in enumerate(zip(docs_raw, metas, distances, ids)):
                docs.append({'id': id_, 'content': doc, 'metadata': meta, 'distance': dist, 'rank': i+1})
        return docs


def rag_simple(query, retriever, llm=None, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([r['content'] for r in results]) if results else ""
    if not context:
        return "No relevant context found."

    if llm is None:
        return f"(No LLM) Context found: \n\n{context[:400]}"

    # llm should provide an invoke([...]) that returns .content
    from langchain_core.messages.human import HumanMessage
    prompt = f"Use the context to answer:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, 'content', str(resp))


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory containing PDFs')
    parser.add_argument('--persist-dir', default='data/vector_store', help='ChromaDB persist directory')
    parser.add_argument('--model', default=os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant'))
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM invocation (safe for CI)')
    args = parser.parse_args()

    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key and not args.skip_llm:
        print("Warning: GROQ_API_KEY not set. Use --skip-llm to run without LLM or set the env var.")

    print("Processing PDFs...")
    docs = process_all_pdfs(args.data_dir)
    print(f"Loaded {len(docs)} pages")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Generating embeddings...")
    em = EmbeddingManager()
    texts = [d.page_content for d in chunks]
    embeddings = em.generate_embeddings(texts)

    print("Persisting to vector store...")
    vs = VectorStore(persist_directory=args.persist_dir)
    vs.add_documents(chunks, embeddings)

    retr = RAGRetriever(vs, em)

    if args.skip_llm:
        print(rag_simple("What is Machine Learning?", retr, llm=None, top_k=args.top_k))
    else:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages.human import HumanMessage
            llm = ChatGroq(groq_api_key=groq_key, model_name=args.model, temperature=0.1, max_tokens=1024)
            print("LLM initialized; running RAG query...")
            print(rag_simple("What is Machine Learning?", retr, llm, top_k=args.top_k)[:2000])
        except Exception as e:
            print("LLM invocation failed:", e)


if __name__ == '__main__':
    main()
