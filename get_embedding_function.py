from typing import Literal
import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


def get_embedding_function(provider: Literal["openai", "hf"] = "hf"):
    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" and api_key:
        return OpenAIEmbeddings()
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)
