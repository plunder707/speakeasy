#!/usr/bin/env python3
"""
Agentic RAG Prototype - Self-Directed Research

The LLM can request additional searches during reasoning.
"""

import re
from wiki_governor import (
    fetch_wikipedia_full,
    fetch_arxiv_full,
    fetch_web_full,
    VectorKnowledgeBase
)
from resonance_reasoner import ResonanceReasoner


class AgenticRAG:
    """
    Self-directed RAG where the LLM can request additional searches.

    Flow:
    1. Initial broad search
    2. LLM evaluates what it has
    3. LLM can request: SEARCH_WIKIPEDIA[query], SEARCH_ARXIV[query], SEARCH_WEB[query]
    4. System executes and adds to context
    5. LLM continues until satisfied
    """

    def __init__(self, max_search_rounds=3):
        self.kb = VectorKnowledgeBase()
        self.max_search_rounds = max_search_rounds
        self.search_history = []

    def run(self, user_query: str, session_id: str = "agentic_session"):
        """
        Run agentic RAG with iterative search.
        """
        print(f"\n🤖 Agentic RAG: Self-Directed Research")
        print(f"=" * 60)
        print(f"Query: {user_query}\n")

        # Round 1: Initial broad search
        print("📚 Round 1: Initial broad search...")
        all_documents = []

        # Wikipedia
        wiki_docs = fetch_wikipedia_full(user_query)
        all_documents.extend(wiki_docs)
        print(f"   ✓ Wikipedia: {len(wiki_docs)} articles")

        # Build initial knowledge base
        if all_documents:
            self.kb.add_documents(all_documents, session_id)
            context = self.kb.retrieve(user_query, session_id, top_k=20)
        else:
            context = ""

        # Iterative search rounds
        for round_num in range(2, self.max_search_rounds + 2):
            print(f"\n🧠 Round {round_num}: LLM evaluating knowledge gaps...")

            # Ask LLM what it needs
            gap_prompt = f"""
You are researching: {user_query}

Current context length: {len(context)} chars

Evaluate what information you're MISSING to answer this query completely.

If you need more information, output EXACTLY ONE of:
- SEARCH_WIKIPEDIA[specific topic]
- SEARCH_ARXIV[specific research topic]
- SEARCH_WEB[specific query]
- READY_TO_ANSWER (if you have enough)

Be specific with search queries. Examples:
- SEARCH_WIKIPEDIA[Douady-Hubbard theorem]
- SEARCH_ARXIV[fractal dimension Mandelbrot set]
- SEARCH_WEB[Julia set visualization examples]

Output ONLY the search command or READY_TO_ANSWER.
"""

            # Create a simple reasoner to evaluate
            reasoner = ResonanceReasoner(
                context,
                model="deepseek-r1:8b",
                enable_multipath=False,
                enable_critique=False
            )

            # Get LLM decision
            decision = ""
            for chunk in reasoner.reason_with_backtracking(gap_prompt, stream=False):
                decision += chunk

            print(f"   LLM Decision: {decision[:200]}...")

            # Parse decision
            if "READY_TO_ANSWER" in decision:
                print("\n✅ LLM has enough information. Generating final answer...")
                break

            # Extract search command
            search_match = re.search(r'SEARCH_(WIKIPEDIA|ARXIV|WEB)\[(.*?)\]', decision)
            if not search_match:
                print(f"\n⚠️  Couldn't parse search command, stopping.")
                break

            search_type = search_match.group(1)
            search_query = search_match.group(2)

            print(f"\n🔍 Executing: SEARCH_{search_type}[{search_query}]")

            # Execute search
            new_docs = []
            if search_type == "WIKIPEDIA":
                new_docs = fetch_wikipedia_full(search_query)
            elif search_type == "ARXIV":
                new_docs = fetch_arxiv_full(search_query)
            elif search_type == "WEB":
                new_docs, _ = fetch_web_full(search_query, session_id)

            if new_docs:
                print(f"   ✓ Retrieved {len(new_docs)} new documents")
                self.kb.add_documents(new_docs, session_id)
                # Re-retrieve with updated knowledge base
                context = self.kb.retrieve(user_query, session_id, top_k=30)
                self.search_history.append({
                    "round": round_num,
                    "type": search_type,
                    "query": search_query,
                    "results": len(new_docs)
                })
            else:
                print(f"   ⚠️  No new documents found")

        # Generate final answer
        print(f"\n" + "=" * 60)
        print(f"🎯 Generating Final Answer...")
        print(f"=" * 60 + "\n")

        final_reasoner = ResonanceReasoner(
            context,
            model="deepseek-r1:8b",
            enable_multipath=True,
            enable_critique=True
        )

        response = ""
        for chunk in final_reasoner.reason_with_backtracking(user_query, stream=True):
            print(chunk, end="", flush=True)
            response += chunk

        # Print search summary
        print(f"\n\n" + "=" * 60)
        print(f"📊 Search Summary:")
        print(f"=" * 60)
        print(f"Total search rounds: {len(self.search_history) + 1}")
        for search in self.search_history:
            print(f"  Round {search['round']}: {search['type']}[{search['query']}] → {search['results']} docs")

        return response


if __name__ == "__main__":
    # Example usage
    rag = AgenticRAG(max_search_rounds=3)

    query = "Explain the Douady-Hubbard theorem connecting Mandelbrot and Julia sets"

    response = rag.run(query)

    print("\n\n✓ Agentic RAG complete!")
