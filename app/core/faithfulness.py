from typing import List
from llama_index.core.schema import NodeWithScore
from app.core.llm import llm

VERIFICATION_PROMPT = """
Task: Verify if the following claim is strictly supported by the provided regulatory context.
Context:
{context_str}

Claim:
{claim}

Is this claim supported? Output ONLY 'YES' or 'NO'.
"""

def split_into_claims(answer: str) -> List[str]:
    """
    Splits the answer into simple atomic claims. 
    In a real system, this would use an LLM. Here we use a simpler split for speed, 
    but with LLM-based verification.
    """
    # Simple split by sentence for demonstration. 
    # For staff-level, we should ideally use an LLM or a robust parser.
    return [s.strip() for s in answer.split('.') if s.strip()]

async def verify_faithfulness(answer: str, nodes: List[NodeWithScore]) -> tuple[bool, float]:
    """
    Verifies that every claim in the answer is grounded in the nodes.
    Returns (is_faithful, score)
    """
    claims = split_into_claims(answer)
    if not claims:
        return True, 1.0

    context_str = "\n\n".join([n.node.get_content() for n in nodes])
    
    supported_count = 0
    for claim in claims:
        prompt = VERIFICATION_PROMPT.format(context_str=context_str, claim=claim)
        response = await llm.acomplete(prompt)
        if "YES" in response.text.upper():
            supported_count += 1
            
    score = supported_count / len(claims)
    # If ANY claim is unsupported, we reject the answer (per instructions)
    is_faithful = (score == 1.0)
    
    return is_faithful, score
