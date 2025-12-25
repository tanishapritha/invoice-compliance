from app.ingestion.index import ingestion_manager

print(f"Total Nodes: {len(ingestion_manager.nodes)}")
if len(ingestion_manager.nodes) > 0:
    print("Sample Node Content:")
    print(ingestion_manager.nodes[0].text[:200])
    
    # Try local retrieval
    retriever = ingestion_manager.get_keyword_retriever()
    if retriever:
        results = retriever.retrieve("registration")
        print(f"Retrieval Results for 'registration': {len(results)}")
        for r in results:
            print(f" - {r.score}: {r.node.text[:50]}...")
    else:
        print("Retriever is None")
else:
    print("No nodes found.")
