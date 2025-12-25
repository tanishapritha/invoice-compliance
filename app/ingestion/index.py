import os
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from app.core.config import config
from app.core.llm import llm

PERSIST_DIR = "./storage"

class IngestionManager:
    def __init__(self):
        self.nodes = []
        self.index = None
        self._initialize()

    def _initialize(self):
        if os.path.exists(PERSIST_DIR):
            try:
                print("Loading existing index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                self.index = load_index_from_storage(storage_context)
                self.nodes = self.index.docstore.docs.values()
                print(f"Loaded index with {len(self.nodes)} nodes")
                return
            except Exception as e:
                print(f"Failed to load index: {e}. Building new index...")

        if not os.path.exists(config.CORPUS_DIR) or not os.listdir(config.CORPUS_DIR):
            print("Corpus directory is empty. Index will be empty.")
            return

        documents = SimpleDirectoryReader(str(config.CORPUS_DIR)).load_data()
        
        parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        try:
            self.nodes = parser.get_nodes_from_documents(documents)
            
            for i, node in enumerate(self.nodes):
                node.metadata["clause_id"] = f"clause_{i}"
                node.metadata["source"] = node.metadata.get("file_name", "unknown")

            print(f"Creating index with {len(self.nodes)} nodes...")
            self.index = VectorStoreIndex(self.nodes)
            
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            print("Index created and persisted successfully")
            
        except Exception as e:
            print(f"CRITICAL: Failed to initialize VectorStoreIndex: {e}")
            print("This is likely due to OpenAI API rate limits.")
            print("Falling back to Keyword Only mode (BM25).")
            # Do NOT clear self.nodes here, so BM25 still works!
            self.index = None
        
    def get_vector_retriever(self, similarity_top_k=3):
        if not self.index:
            return None
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
        )

    def get_keyword_retriever(self, similarity_top_k=3):
        if not self.nodes:
            return None
        return BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=similarity_top_k
        )

ingestion_manager = IngestionManager()
