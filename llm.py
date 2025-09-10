import os
import sys
from langchain_groq import ChatGroq
from openai import OpenAI
import google.generativeai as genai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, DEFAULT_MODEL


def get_openai_model(model: str = DEFAULT_MODEL):
    try:
        api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")


def get_chatgroq_model(model: str = "llama-3.1-8b-instant"):
    try:
        api_key = os.getenv("GROQ_API_KEY")  
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment variables")
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


def get_gemini_model(model: str = "gemini-1.5-pro"):
    try:
        api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        model_obj = genai.GenerativeModel(model)
        return model_obj
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")


def load_llm(provider: str, model: str = DEFAULT_MODEL):
    provider = provider.lower()
    if provider == "openai":
        return get_openai_model(model)
    elif provider == "groq":
        return get_chatgroq_model(model)
    elif provider == "gemini":
        return get_gemini_model(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
