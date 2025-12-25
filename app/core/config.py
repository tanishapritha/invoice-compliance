import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
        raise ValueError("FATAL: Both OPENAI_API_KEY and OPENROUTER_API_KEY are missing.")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CORPUS_DIR = BASE_DIR / "corpus"
    LLAMA_INDEX_CACHE_DIR = os.getenv("LLAMA_INDEX_CACHE_DIR", str(BASE_DIR / ".llama_index_cache"))
    
    # Ensure directories exist
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    Path(LLAMA_INDEX_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # LlamaIndex Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

config = Config()
