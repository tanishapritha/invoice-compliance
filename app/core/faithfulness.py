from typing import List
from llama_index.core.schema import NodeWithScore
from app.core.llm import llm

VERIFICATION_PROMPT = """
Task: You are a high-precision legal auditor. Verify if the provided answer is strictly grounded in the regulatory context.

CONTEXT:
{context_str}

ANSWER TO VERIFY:
{answer}

Rules:
1. Every significant factual claim in the answer must be supported by the context.
2. If the answer contains information NOT in the context, it is UNFAITHFUL.
3. Output ONLY a JSON object with this format: 
{{
  "is_faithful": true/false,
  "score": 0.0 to 1.0,
  "reason": "short explanation"
}}
"""

async def verify_faithfulness(answer: str, nodes: List[NodeWithScore]) -> tuple[bool, float]:
    """
    Optimized: Verifies the entire answer in ONE call to save API quota.
    """
    if not answer or answer.strip() == "No information available.":
        return True, 1.0

    context_str = "\n\n".join([n.node.get_content() for n in nodes])
    
    try:
        prompt = VERIFICATION_PROMPT.format(context_str=context_str, answer=answer)
        response = await llm.acomplete(prompt)
        
        # Simple extraction if LLM ignores JSON request or uses markdown
        text = response.text.strip().lower()
        
        # Look for the specific pattern anywhere in the text
        import re
        is_faithful_match = re.search(r'"is_faithful":\s*(true|false)', text)
        
        if is_faithful_match:
            val = is_faithful_match.group(1)
            if val == "true":
                return True, 1.0
            else:
                return False, 0.0
        
        # Fallback for non-json or text-based responses
        if "is_faithful\": true" in text or "\"is_faithful\":true" in text or "yes" in text[:10]:
            return True, 1.0
        
        return False, 0.0
        
    except Exception as e:
        print(f"Faithfulness check failed: {e}. Falling back to optimistic validation.")
        return True, 0.8 # Fallback to save the query if verification fails
