import uuid
from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    QueryRequest, AnswerResponse, AbstainResponse, 
    RetrievalNode, ConfidenceLevel, Outcome
)
from app.core.retrieval import hybrid_retrieve
from app.core.confidence import calculate_confidence
from app.core.abstention import should_abstain, generate_abstain_response
from app.core.generation import generate_answer
from app.core.faithfulness import verify_faithfulness
from app.core.audit_logger import log_query, get_logs
from app.ingestion.index import ingestion_manager

router = APIRouter()

@router.post("/query", response_model=None)
async def query_compliance(request: QueryRequest):
    query_id = str(uuid.uuid4())
    
    # 1. Retrieval
    nodes = await hybrid_retrieve(request.question)
    
    # Needs individual results for confidence scoring
    vector_retriever = ingestion_manager.get_vector_retriever()
    keyword_retriever = ingestion_manager.get_keyword_retriever()
    v_results = vector_retriever.retrieve(request.question) if vector_retriever else []
    k_results = keyword_retriever.retrieve(request.question) if keyword_retriever else []

    # 2. Confidence Scoring
    confidence = calculate_confidence(request.question, nodes, v_results, k_results)
    
    # 3. Abstention Gate
    abstain, reason = should_abstain(confidence, len(nodes))
    if abstain:
        log_query(query_id, request.question, request.jurisdiction, Outcome.ABSTAINED, confidence)
        return generate_abstain_response(query_id, reason)
    
    # 4. Generation
    answer = await generate_answer(request.question, nodes)
    
    # 5. Faithfulness Verification
    is_faithful, faith_score = await verify_faithfulness(answer, nodes)
    if not is_faithful:
        log_query(query_id, request.question, request.jurisdiction, Outcome.ABSTAINED, confidence)
        return generate_abstain_response(query_id, "Generated answer failed internal faithfulness verification and was rejected to prevent hallucination.")
    
    # 6. Audit Logging
    log_query(query_id, request.question, request.jurisdiction, Outcome.ANSWERED, confidence)
    
    # Mapping nodes for response
    grounding_nodes = [
        RetrievalNode(
            id=n.node.node_id,
            text=n.node.get_content(),
            score=n.score or 0.0,
            metadata=n.node.metadata
        ) for n in nodes
    ]
    
    return AnswerResponse(
        query_id=query_id,
        answer=answer,
        confidence=confidence,
        grounding_nodes=grounding_nodes,
        faithfulness_score=faith_score
    )

@router.post("/debug/retrieval")
async def debug_retrieval(request: QueryRequest):
    nodes = await hybrid_retrieve(request.question)
    return [{ "id": n.node.node_id, "text": n.node.get_content(), "score": n.score } for n in nodes]

@router.post("/debug/faithfulness")
async def debug_faithfulness(query: str, answer: str):
    nodes = await hybrid_retrieve(query)
    is_faithful, score = await verify_faithfulness(answer, nodes)
    return { "is_faithful": is_faithful, "score": score }

@router.get("/audit/logs")
async def audit_logs():
    return get_logs()
