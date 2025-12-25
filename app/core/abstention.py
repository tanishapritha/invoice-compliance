from app.schemas.models import ConfidenceLevel, AbstainResponse, Outcome

def should_abstain(confidence: ConfidenceLevel, nodes_count: int) -> tuple[bool, str]:
    """
    Decides whether to abstain from generating an answer.
    Returns (should_abstain, reason)
    """
    if nodes_count == 0:
        return True, "No relevant regulatory clauses found in the corpus."
    
    if confidence == ConfidenceLevel.LOW:
        return True, "Confidence in the retrieved information is too low to provide a safe answer."
    
    return False, ""

def generate_abstain_response(query_id: str, reason: str, confidence: ConfidenceLevel = ConfidenceLevel.LOW) -> AbstainResponse:
    return AbstainResponse(
        query_id=query_id,
        reason=reason,
        confidence=confidence,
        outcome=Outcome.ABSTAINED
    )
