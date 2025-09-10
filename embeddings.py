import os
import sys
from typing import List

from openai import OpenAI
import google.generativeai as genai

# Optional: for local embeddings (offline mode)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import OPENAI_API_KEY, GEMINI_API_KEY


def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in response.data]
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding failed: {str(e)}")


def get_gemini_embeddings(texts: List[str], model: str = "models/embedding-001") -> List[List[float]]:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        embedder = genai.embed_content(model=model, content=texts)
        return embedder["embedding"]
    except Exception as e:
        raise RuntimeError(f"Gemini embedding failed: {str(e)}")


def get_local_embeddings(texts: List[str], model: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    try:
        embedder = SentenceTransformer(model)
        return embedder.encode(texts).tolist()
    except Exception as e:
        raise RuntimeError(f"Local embedding failed: {str(e)}")


def get_embeddings(provider: str, texts: List[str]) -> List[List[float]]:
    provider = provider.lower()
    if provider == "openai":
        return get_openai_embeddings(texts)
    elif provider == "gemini":
        return get_gemini_embeddings(texts)
    elif provider == "local":
        return get_local_embeddings(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
