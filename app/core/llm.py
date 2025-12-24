from llama_index.llms.openai import OpenAI
from app.core.config import config

def get_llm():
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        model="gpt-4-turbo-preview"
    )

llm = get_llm()
