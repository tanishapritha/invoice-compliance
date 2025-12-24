from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Jurisdiction(str, Enum):
    DPDP = "DPDP"
    GDPR = "GDPR"

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Outcome(str, Enum):
    ANSWERED = "ANSWERED"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"

class QueryRequest(BaseModel):
    question: str
    jurisdiction: Jurisdiction

class RetrievalNode(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class AnswerResponse(BaseModel):
    query_id: str
    answer: str
    confidence: ConfidenceLevel
    grounding_nodes: List[RetrievalNode]
    faithfulness_score: float
    outcome: Outcome = Outcome.ANSWERED

class AbstainResponse(BaseModel):
    query_id: str
    reason: str
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    outcome: Outcome = Outcome.ABSTAINED

class AuditLogEntry(BaseModel):
    query_id: str
    question: str
    jurisdiction: Jurisdiction
    outcome: Outcome
    confidence_level: ConfidenceLevel
    timestamp: datetime
