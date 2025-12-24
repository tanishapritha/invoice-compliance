from typing import List
from llama_index.core.schema import NodeWithScore
from app.core.config import config
from app.core.llm import llm

GENERATION_PROMPT = """
You are a strict regulatory compliance assistant. 
Use ONLY the provided clauses below to answer the user's question.
If the clauses do not contain the answer, you MUST state that you do not know.
DO NOT use any external knowledge.
DO NOT paraphrase in a way that changes the legal meaning.

CLAUSES:
{context_str}

QUESTION:
{question}

ANSWER:
"""

async def generate_answer(question: str, nodes: List[NodeWithScore]) -> str:
    """
    Generates an answer constrained strictly to the provided nodes.
    """
    if not nodes:
        return "No information available."

    context_str = "\n\n".join([
        f"Clause ID: {n.node.metadata.get('clause_id', 'unknown')}\n{n.node.get_content()}" 
        for n in nodes
    ])
    
    prompt = GENERATION_PROMPT.format(context_str=context_str, question=question)
    
    response = await llm.acomplete(prompt)
    return response.text.strip()
