
import os
import sys
import faiss
import pickle
import logging
import numpy as np
from typing import List, Tuple
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from embeddings import get_embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Document Loading
def load_documents(file_path: str):
    try:
        _, extension = os.path.splitext(file_path)
        
        if extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension.lower() == ".txt":
            loader = TextLoader(file_path)
        else:
            logging.warning(f"Unsupported file type '{extension}' for file: {os.path.basename(file_path)}. Skipping.")
            return [] 
            
        # Load the document
        documents = loader.load()
        return documents
        
    except Exception as e:
        logging.error(f"❌ Error loading document {os.path.basename(file_path)}: {e}")
        return [] # Return empty list on error

def load_and_index_docs(uploaded_files, provider: str = "openai", save_path: str = "data/vector_store.pkl"):
    all_chunks = []
    os.makedirs("data", exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            # Save file locally
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load document (PDF or TXT)
            docs = load_documents(file_path)

            # Split into chunks
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

        except Exception as e:
            logging.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

    if all_chunks:
        return build_vector_store(all_chunks, provider, save_path)
    else:
        return None, []
    
# Text Chunking
def chunk_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)
    except Exception as e:
        logging.error(f"Error splitting documents: {str(e)}")
        return []

# Vector Store (FAISS)
def build_vector_store(chunks, provider: str = "openai", save_path: str = "data/vector_store.pkl"):
    try:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = get_embeddings(provider, texts)

        embeddings_array = np.array(embeddings).astype("float32")

        # Convert to FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # Save index + mapping
        with open(save_path, "wb") as f:
            pickle.dump({"index": index, "chunks": chunks}, f)

        return index, chunks
    except Exception as e:
        logging.error(f"Error building vector store: {str(e)}")
        return None, []


def load_vector_store(path: str = "data/vector_store.pkl"):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["index"], data["chunks"]
    except Exception as e:
        logging.error(f"Error loading vector store: {str(e)}")
        return None, []


def retrieve(query: str, index, chunks, provider: str = "openai", k: int = 8) -> List[str]:
    try:
        query_emb = get_embeddings(provider, [query])
        query_emb_array = np.array(query_emb).astype("float32")
        distances, indices = index.search(query_emb_array, k)
        results = [chunks[i].page_content for i in indices[0] if i < len(chunks)]
        return results
    except Exception as e:
        logging.error(f"Error retrieving documents: {str(e)}")
        return []
    

retrieve_relevant_chunks = retrieve

def perform_web_search(query: str) -> str:
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        
        # Format the results into a single, clean string for the LLM
        formatted_results = "\n\n".join(
            [f"Source: {res['url']}\nContent: {res['content']}" for res in results]
        )
        return formatted_results
    except Exception as e:
        logging.error(f"Web search failed: {str(e)}")
        return f"Sorry, the web search failed with the following error: {str(e)}"

