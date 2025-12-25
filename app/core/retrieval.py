from typing import List
from llama_index.core.schema import NodeWithScore
from app.ingestion.index import ingestion_manager

async def hybrid_retrieve(query: str) -> List[NodeWithScore]:
    """
    Retreives nodes from both vector and keyword retrievers and merges them.
    Explicitly handles empty indices by returning an empty list.
    """
    vector_retriever = ingestion_manager.get_vector_retriever()
    keyword_retriever = ingestion_manager.get_keyword_retriever()

    if not vector_retriever and not keyword_retriever:
        return []

    # Get results from available retrievers
    vector_results = vector_retriever.retrieve(query) if vector_retriever else []
    keyword_results = keyword_retriever.retrieve(query) if keyword_retriever else []

    # Merge results (simple union by node ID)
    seen_ids = set()
    merged = []
    
    for res in vector_results + keyword_results:
        if res.node.node_id not in seen_ids:
            merged.append(res)
            seen_ids.add(res.node.node_id)
            
    return merged
