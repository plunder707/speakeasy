#!/usr/bin/env python3
"""Quick test of agentic RAG functionality"""

from agentic_rag import AgenticRAG

print("=" * 60)
print("Testing Agentic RAG - Quick Validation")
print("=" * 60)

# Create agentic RAG instance
rag = AgenticRAG(max_search_rounds=2, enable_streaming=False)

# Simple test query
query = "What is quantum entanglement?"

print(f"\nQuery: {query}\n")

try:
    response, metadata = rag.run(query, session_id="test_session")

    print("\n" + "=" * 60)
    print("✅ TEST PASSED")
    print("=" * 60)
    print(f"Search rounds: {metadata['total_search_rounds']}")
    print(f"Documents: {metadata['total_documents']}")
    print(f"Context size: {metadata['final_context_size']:,} chars")
    print(f"Final drift: {metadata['final_drift']:.3f}")
    print(f"\nResponse length: {len(response)} chars")
    print(f"\nFirst 200 chars of response:")
    print(response[:200] + "...")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ TEST FAILED")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
