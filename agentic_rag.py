#!/usr/bin/env python3
"""
Agentic RAG - Production Version with Enhanced Decision Making

Features:
- Structured JSON output for reliable command parsing
- Search history tracking to avoid loops
- Confidence scoring for decisions
- Streaming support for UI integration
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from wiki_governor import (
    fetch_wikipedia_full,
    fetch_arxiv_full,
    fetch_web_full,
    VectorKnowledgeBase
)
from resonance_reasoner import ResonanceReasoner


@dataclass
class SearchAction:
    """Structured search decision from LLM"""
    action: str  # "search_wikipedia", "search_arxiv", "search_web", "answer"
    query: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


class AgenticRAG:
    """
    Production agentic RAG with autonomous search decisions.

    Enhancements over prototype:
    1. Structured JSON output for reliable parsing
    2. Search history context to prevent loops
    3. Confidence scoring for decision validation
    4. Streaming support for real-time UI updates
    """

    def __init__(self, max_search_rounds=5, enable_streaming=True):
        self.kb = VectorKnowledgeBase()
        self.max_search_rounds = max_search_rounds
        self.enable_streaming = enable_streaming
        self.search_history = []

    def _parse_llm_decision(self, decision_text: str) -> SearchAction:
        """
        Parse LLM decision with fallback to regex if JSON fails.

        Tries:
        1. JSON extraction: {"action": "search_wikipedia", "query": "...", "confidence": 0.9}
        2. Regex fallback: SEARCH_WIKIPEDIA[query]
        3. Ready check: READY_TO_ANSWER
        """
        # Try JSON extraction first
        json_match = re.search(r'\{[^}]*"action"[^}]*\}', decision_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return SearchAction(
                    action=data.get("action", "answer"),
                    query=data.get("query"),
                    confidence=data.get("confidence", 0.0),
                    reasoning=data.get("reasoning", "")
                )
            except json.JSONDecodeError:
                pass

        # Fallback to regex parsing
        if "READY_TO_ANSWER" in decision_text:
            # Extract confidence if present
            conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', decision_text, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.95

            return SearchAction(
                action="answer",
                confidence=confidence,
                reasoning="LLM indicated sufficient information"
            )

        search_match = re.search(r'SEARCH_(WIKIPEDIA|ARXIV|WEB)\[(.*?)\]', decision_text)
        if search_match:
            search_type = search_match.group(1).lower()
            search_query = search_match.group(2)

            # Extract confidence if present
            conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', decision_text, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.7

            return SearchAction(
                action=f"search_{search_type}",
                query=search_query,
                confidence=confidence,
                reasoning=f"Need more information about: {search_query}"
            )

        # Default: assume ready to answer
        return SearchAction(
            action="answer",
            confidence=0.5,
            reasoning="Could not parse decision, proceeding with answer"
        )

    def _should_search(self, query: str) -> bool:
        """
        Determine if query needs external research or can use model knowledge.

        Returns False for:
        - Code writing requests
        - General conversation
        - Math/logic problems
        - "What do you think" questions
        """
        query_lower = query.lower()

        # Code-related queries - don't need Wikipedia
        code_keywords = ['write code', 'write a function', 'write a program', 'implement',
                        'debug', 'fix this code', 'python code', 'javascript', 'function that',
                        'create a class', 'algorithm for', 'regex for']
        if any(kw in query_lower for kw in code_keywords):
            return False

        # Conversational queries
        convo_keywords = ['what do you think', 'your opinion', 'tell me about yourself',
                         'how are you', 'can you help', 'joke', 'story']
        if any(kw in query_lower for kw in convo_keywords):
            return False

        # Math/logic - model can handle directly
        if any(op in query for op in ['+', '-', '*', '/', '=', '<', '>']):
            return False

        # Everything else likely benefits from research
        return True

    def _format_search_history(self) -> str:
        """Format search history for context"""
        if not self.search_history:
            return "None yet"

        history_lines = []
        for search in self.search_history:
            history_lines.append(
                f"  - Round {search['round']}: {search['type'].upper()}[{search['query']}] "
                f"→ {search['results']} docs (confidence: {search.get('confidence', 'N/A')})"
            )
        return "\n".join(history_lines)

    async def run(
        self,
        user_query: str,
        session_id: str = "agentic_session",
        status_callback=None
    ) -> Tuple[str, Dict]:
        """
        Run agentic RAG with iterative search and streaming support.

        Args:
            user_query: User's research question
            session_id: Session identifier for vector DB
            status_callback: Optional async callback(message: str) for UI updates

        Returns:
            (final_answer, metadata_dict)
        """
        async def log_status(msg: str):
            """Log status to callback or print"""
            if status_callback:
                import asyncio
                import inspect
                # Check if callback is async
                if inspect.iscoroutinefunction(status_callback):
                    await status_callback(msg)
                else:
                    status_callback(msg)
            else:
                print(msg)

        await log_status(f"🤖 Agentic RAG: Autonomous Research Mode")
        await log_status(f"Query: {user_query}\n")

        # Pre-check: Does this query actually need research?
        # Some queries are conversational, code requests, or don't need external sources
        needs_research = self._should_search(user_query)

        if not needs_research:
            await log_status(f"💬 Query doesn't require external research - using model's knowledge directly\n")
            # Skip search, go straight to generation with minimal context
            context = "No external sources needed for this query."
            all_documents = []
        else:
            # Round 1: Initial broad search
            await log_status("📚 Round 1: Initial broad search...")
            all_documents = []

            # Wikipedia initial search
            wiki_docs = fetch_wikipedia_full(user_query)
            all_documents.extend(wiki_docs)
            await log_status(f"   ✓ Wikipedia: {len(wiki_docs)} articles")

            # arXiv initial search
            arxiv_docs = fetch_arxiv_full(user_query)
            all_documents.extend(arxiv_docs)
            await log_status(f"   ✓ arXiv: {len(arxiv_docs)} papers")

            # Web initial search
            web_docs, _ = fetch_web_full(user_query, session_id)
            all_documents.extend(web_docs)
            await log_status(f"   ✓ Web: {len(web_docs)} pages")

            # Build initial knowledge base
            if all_documents:
                self.kb.add_documents(all_documents, session_id)
                context = self.kb.retrieve(user_query, session_id, top_k=20)
            else:
                context = ""

            # Iterative search rounds
            for round_num in range(2, self.max_search_rounds + 2):
                await log_status(f"\n🧠 Round {round_num}: Evaluating knowledge gaps...")

                # Enhanced gap analysis prompt with structured output
                gap_prompt = f"""You are researching: {user_query}

    Current status:
    - Context available: {len(context)} characters
    - Documents retrieved: {len(all_documents)}
    - Search history:
    {self._format_search_history()}

    Evaluate if you have ENOUGH information to answer the query completely.

    Output your decision in JSON format:
    {{
    "action": "search_wikipedia" | "search_arxiv" | "search_web" | "answer",
    "query": "specific search query (if searching)",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of decision"
    }}

    Alternatively, you can use the legacy format:
    - SEARCH_WIKIPEDIA[specific topic] (confidence: 0.X)
    - SEARCH_ARXIV[specific research topic] (confidence: 0.X)
    - SEARCH_WEB[specific query] (confidence: 0.X)
    - READY_TO_ANSWER (confidence: 0.X)

    Guidelines:
    - Only search if you're missing CRITICAL information (confidence < 0.8 for answer)
    - Avoid repeating previous searches
    - Be specific with queries
    - Consider if existing context is sufficient before requesting more
    """

                # Create reasoner for gap evaluation (fast, no multipath)
                reasoner = ResonanceReasoner(
                    context,
                    model="deepseek-r1:8b",
                    enable_multipath=True,
                    enable_critique=True
                )

                # Get LLM decision
                decision_text = ""
                async for chunk in reasoner.reason_with_backtracking(gap_prompt, stream=False):
                    decision_text += chunk

                # Parse decision
                decision = self._parse_llm_decision(decision_text)

                await log_status(f"   Decision: {decision.action} (confidence: {decision.confidence:.2f})")
                if decision.reasoning:
                    await log_status(f"   Reasoning: {decision.reasoning[:100]}...")

                # Check if ready to answer
                if decision.action == "answer":
                    await log_status(f"\n✅ LLM ready to answer (confidence: {decision.confidence:.2f})")
                    break

                # Validate confidence threshold
                if decision.confidence < 0.5:
                    await log_status(f"   ⚠️ Low confidence ({decision.confidence:.2f}), proceeding anyway...")

                # Extract search parameters
                if not decision.query:
                    await log_status(f"\n⚠️ No search query specified, stopping.")
                    break

                # Check for repeated searches
                if any(s['query'].lower() == decision.query.lower() for s in self.search_history):
                    await log_status(f"\n⚠️ Already searched for '{decision.query}', stopping to avoid loop.")
                    break

                await log_status(f"\n🔍 Executing: {decision.action.upper()}[{decision.query}]")

                # Execute search
                new_docs = []
                search_type = decision.action.replace("search_", "")

                if search_type == "wikipedia":
                    new_docs = fetch_wikipedia_full(decision.query)
                elif search_type == "arxiv":
                    new_docs = fetch_arxiv_full(decision.query)
                elif search_type == "web":
                    new_docs, _ = fetch_web_full(decision.query, session_id)

                if new_docs:
                    await log_status(f"   ✓ Retrieved {len(new_docs)} new documents")
                    self.kb.add_documents(new_docs, session_id)
                    all_documents.extend(new_docs)

                    # Re-retrieve with updated knowledge base
                    context = self.kb.retrieve(user_query, session_id, top_k=30)

                    # Record search in history
                    self.search_history.append({
                        "round": round_num,
                        "type": search_type,
                        "query": decision.query,
                        "results": len(new_docs),
                        "confidence": decision.confidence
                    })
                else:
                    await log_status(f"   ⚠️ No new documents found")

        # Generate final answer with full reasoning
        await log_status(f"\n{'=' * 60}")
        await log_status(f"🎯 Generating Final Answer...")
        await log_status(f"{'=' * 60}\n")

        final_reasoner = ResonanceReasoner(
            context,
            model="deepseek-r1:8b",
            enable_multipath=True,
            enable_critique=True,
            drift_abort_threshold=0.95
        )

        response = ""
        if self.enable_streaming:
            async for chunk in final_reasoner.reason_with_backtracking(user_query, stream=True):
                if status_callback:
                    import inspect
                    # Properly await async streaming callback
                    if inspect.iscoroutinefunction(status_callback):
                        await status_callback(chunk, is_stream=True)
                    else:
                        status_callback(chunk, is_stream=True)
                else:
                    print(chunk, end="", flush=True)
                response += chunk
        else:
            async for chunk in final_reasoner.reason_with_backtracking(user_query, stream=False):
                response += chunk

        # Compile metadata
        metadata = {
            "total_search_rounds": len(self.search_history) + 1,
            "search_history": self.search_history,
            "total_documents": len(all_documents),
            "final_context_size": len(context),
            "final_drift": final_reasoner.current_drift,
            "reasoning_attempts": len(final_reasoner.attempts)
        }

        await log_status(f"\n\n{'=' * 60}")
        await log_status(f"📊 Search Summary:")
        await log_status(f"{'=' * 60}")
        await log_status(f"Total rounds: {metadata['total_search_rounds']}")
        await log_status(f"Documents: {metadata['total_documents']}")
        await log_status(f"Final drift: {metadata['final_drift']:.3f}")
        for search in self.search_history:
            await log_status(
                f"  Round {search['round']}: {search['type'].upper()}[{search['query']}] "
                f"→ {search['results']} docs (conf: {search.get('confidence', 'N/A')})"
            )

        return response, metadata


if __name__ == "__main__":
    # Example usage
    rag = AgenticRAG(max_search_rounds=3, enable_streaming=True)

    query = "What is the relationship between quantum entanglement and Bell's theorem?"

    response, metadata = rag.run(query)

    print("\n\n✅ Agentic RAG complete!")
    print(f"Final drift: {metadata['final_drift']:.3f}")
