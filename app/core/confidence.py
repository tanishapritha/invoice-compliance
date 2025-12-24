import numpy as np
from typing import List
from llama_index.core.schema import NodeWithScore
from app.schemas.models import ConfidenceLevel

def calculate_confidence(
    query: str, 
    retrieved_nodes: List[NodeWithScore],
    vector_results: List[NodeWithScore],
    keyword_results: List[NodeWithScore]
) -> ConfidenceLevel:
    """
    Computes confidence based on score variance, term coverage, and retriever agreement.
    """
    if not retrieved_nodes:
        return ConfidenceLevel.LOW

    # 1. Score Variance (how distinct is the top result?)
    scores = [n.score for n in retrieved_nodes if n.score is not None]
    if len(scores) > 1:
        variance = np.var(scores)
    else:
        variance = 0.0

    # 2. Term Coverage (crude check: do query terms appear in nodes?)
    query_terms = set(query.lower().split())
    combined_text = " ".join([n.node.get_content().lower() for n in retrieved_nodes])
    coverage = sum(1 for term in query_terms if term in combined_text) / len(query_terms) if query_terms else 0

    # 3. Cross-Retriever Agreement
    vector_ids = {n.node.node_id for n in vector_results}
    keyword_ids = {n.node.node_id for n in keyword_results}
    intersection = vector_ids.intersection(keyword_ids)
    agreement = len(intersection) / max(len(vector_ids), len(keyword_ids)) if max(len(vector_ids), len(keyword_ids)) > 0 else 0

    # Heuristic Logic
    confidence_score = (variance * 0.2) + (coverage * 0.5) + (agreement * 0.3)

    if confidence_score > 0.7:
        return ConfidenceLevel.HIGH
    elif confidence_score > 0.4:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
