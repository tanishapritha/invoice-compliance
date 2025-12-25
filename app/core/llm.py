from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from app.core.config import config

def setup_settings():
    # Use OpenRouter for Generation to bypass direct OpenAI quota issues
    Settings.llm = OpenRouter(
        api_key=config.OPENROUTER_API_KEY,
        model="openai/gpt-4o-mini"
    )
    # Use Local HuggingFace for Embeddings to bypass OpenAI Quota issues
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return Settings.llm

llm = setup_settings()
