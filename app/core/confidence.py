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
    top_score = max(scores) if scores else 0
    # Normalize score influence (assuming BM25 or Vector scores are roughly in 0-10 range)
    score_strength = min(top_score / 1.0, 1.0) 

    # 2. Term Coverage (Improved: filtered stop words)
    stop_words = {"what", "are", "the", "for", "is", "a", "an", "does", "do", "of", "in", "on", "to", "with", "and"}
    query_terms = [term for term in query.lower().split() if term not in stop_words]
    
    combined_text = " ".join([n.node.get_content().lower() for n in retrieved_nodes])
    if not query_terms:
        coverage = 1.0
    else:
        coverage = sum(1 for term in query_terms if term in combined_text) / len(query_terms)

    # 3. Cross-Retriever Agreement
    vector_ids = {n.node.node_id for n in vector_results}
    keyword_ids = {n.node.node_id for n in keyword_results}
    intersection = vector_ids.intersection(keyword_ids)
    
    max_len = max(len(vector_ids), len(keyword_ids))
    agreement = len(intersection) / max_len if max_len > 0 else 0

    # Heuristic Logic: Coverage is the most important signal for RAG
    confidence_score = (score_strength * 0.2) + (coverage * 0.6) + (agreement * 0.2)
    
    print(f"DEBUG CONFIDENCE: Score Strength: {score_strength:.2f}, Coverage: {coverage:.2f}, Agreement: {agreement:.2f} -> Total: {confidence_score:.2f}")

    if confidence_score > 0.6:
        return ConfidenceLevel.HIGH
    elif confidence_score > 0.3:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
