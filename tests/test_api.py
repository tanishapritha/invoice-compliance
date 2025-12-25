import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Regulatory Compliance RAG API is running."}

def test_query_endpoint_structure():
    """
    Test that the query endpoint accepts a valid payload and returns a valid response structure.
    Does not strictly assert the content of the answer, as that depends on LLM/Indexing.
    """
    payload = {
        "question": "What are the core principles of the act?",
        "jurisdiction": "DPDP"
    }
    response = client.post("/api/v1/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "query_id" in data
    assert "outcome" in data
    
    if data["outcome"] == "ANSWERED":
        assert "answer" in data
        assert "faithfulness_score" in data
    elif data["outcome"] == "ABSTAINED":
        assert "reason" in data

def test_debug_retrieval():
    payload = {
        "question": "test query",
        "jurisdiction": "DPDP"
    }
    response = client.post("/api/v1/debug/retrieval", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
