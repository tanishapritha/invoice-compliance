import json
import os
from datetime import datetime
from app.schemas.models import AuditLogEntry, Jurisdiction, Outcome, ConfidenceLevel

LOG_FILE = "audit_logs.jsonl"

def log_query(
    query_id: str,
    question: str,
    jurisdiction: Jurisdiction,
    outcome: Outcome,
    confidence_level: ConfidenceLevel
):
    entry = AuditLogEntry(
        query_id=query_id,
        question=question,
        jurisdiction=jurisdiction,
        outcome=outcome,
        confidence_level=confidence_level,
        timestamp=datetime.now()
    )
    
    with open(LOG_FILE, "a") as f:
        f.write(entry.model_dump_json() + "\n")

def get_logs() -> list[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    
    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            logs.append(json.loads(line))
    return logs
