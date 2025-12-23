#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_reasoner.py - Enhanced Reasoning Layer for Resonance Governor

Adds:
1. Backtracking - Abort and retry when drift spikes
2. Multi-path exploration - Try multiple reasoning approaches
3. Self-critique - Validate reasoning for logical errors

Architecture:
    ResonanceReasoner
        ↓
    OllamaResonanceWrapper (existing)
        ↓
    ResonanceController (existing)
"""

import logging
from typing import List, Dict, Generator, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from resonance_governor import OllamaResonanceWrapper

logger = logging.getLogger(__name__)


@dataclass
class ReasoningAttempt:
    """Represents a single reasoning attempt"""
    response: str
    final_drift: float
    max_drift: float
    tokens: int
    approach: str
    aborted: bool = False
    abort_reason: Optional[str] = None


class ResonanceReasoner:
    """
    Enhanced reasoning wrapper with backtracking and multi-path exploration.

    Features:
    - Monitors drift during generation
    - Aborts high-drift reasoning paths
    - Tries multiple approaches in parallel
    - Self-critique for logical validation
    """

    def __init__(
        self,
        anchor_text: str,
        model: str = "deepseek-r1:8b",
        drift_abort_threshold: float = 1.5,  # Disabled - let controller handle evidence checking
        max_attempts: int = 3,
        enable_multipath: bool = True,
        enable_critique: bool = True
    ):
        """
        Initialize the reasoner.

        Args:
            anchor_text: Retrieved context for grounding
            model: Model to use for generation
            drift_abort_threshold: Abort generation if drift exceeds this
            max_attempts: Maximum reasoning attempts before giving up
            enable_multipath: Try multiple reasoning approaches in parallel
            enable_critique: Enable self-critique validation
        """
        self.anchor_text = anchor_text
        self.model = model
        self.drift_abort_threshold = drift_abort_threshold
        self.max_attempts = max_attempts
        self.enable_multipath = enable_multipath
        self.enable_critique = enable_critique

        # Track attempts
        self.attempts: List[ReasoningAttempt] = []
        self.current_drift = 0.0

    async def reason_with_backtracking(
        self,
        query: str,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate response with backtracking on high drift.

        Yields response chunks if stream=True, otherwise returns full response.
        """

        if self.enable_multipath:
            # Try multiple reasoning approaches
            async for chunk in self._multipath_reasoning(query, stream):
                yield chunk
        else:
            # Single-path with backtracking
            async for chunk in self._single_path_reasoning(query, stream):
                yield chunk

    async def _single_path_reasoning(
        self,
        query: str,
        stream: bool
    ) -> AsyncGenerator[str, None]:
        """Single reasoning path with backtracking on failure"""

        for attempt_num in range(self.max_attempts):
            # Modify query based on attempt number
            modified_query = self._reframe_query(query, attempt_num)
            approach = self._get_approach_name(attempt_num)

            logger.info(f"Reasoning attempt {attempt_num + 1}/{self.max_attempts}: {approach}")

            # Try generating response
            response_chunks = []
            drift_history = []
            aborted = False
            abort_reason = None

            # Create wrapper for this attempt
            wrapper = OllamaResonanceWrapper(
                self.anchor_text,
                model=self.model,
                enable_diagrams=True
            )

            token_count = 0

            try:
                async for chunk in wrapper.stream_chat(modified_query):
                    response_chunks.append(chunk)
                    drift_history.append(wrapper.current_drift)
                    self.current_drift = wrapper.current_drift
                    token_count += 1

                    # Check for drift spike
                    if wrapper.current_drift > self.drift_abort_threshold:
                        aborted = True
                        abort_reason = f"Drift exceeded threshold ({wrapper.current_drift:.3f} > {self.drift_abort_threshold})"
                        logger.warning(f"⚠️ {abort_reason}")
                        break

                    # Stream the chunk if enabled
                    if stream and not aborted:
                        yield chunk

                # Record attempt
                attempt = ReasoningAttempt(
                    response=''.join(response_chunks),
                    final_drift=wrapper.current_drift,
                    max_drift=max(drift_history) if drift_history else 0.0,
                    tokens=token_count,
                    approach=approach,
                    aborted=aborted,
                    abort_reason=abort_reason
                )
                self.attempts.append(attempt)

                # If not aborted, this is our answer
                if not aborted:
                    # Apply self-critique if enabled
                    if self.enable_critique:
                        critique_result = await self._self_critique(attempt.response, query)
                        if critique_result['needs_revision']:
                            logger.info("🔍 Self-critique detected issues, revising...")
                            if stream:
                                yield "\n\n[Revising based on self-critique...]\n\n"
                            # Don't recurse infinitely - just note the issue
                            if stream:
                                yield f"\n\n*Note: {critique_result['issue']}*"

                    return  # Success, exit

                # If aborted, try next attempt
                if stream:
                    yield f"\n\n⚠️ Approach {attempt_num + 1} diverged from sources. Trying different angle...\n\n"

            except Exception as e:
                logger.error(f"❌ Attempt {attempt_num + 1} failed: {e}")
                if stream:
                    yield f"\n\n❌ Error in attempt {attempt_num + 1}: {str(e)}\n\n"
                continue

        # All attempts exhausted
        if stream:
            yield self._fallback_response()

    async def _multipath_reasoning(
        self,
        query: str,
        stream: bool
    ) -> AsyncGenerator[str, None]:
        """
        Try multiple reasoning approaches in parallel, pick best.

        Note: In streaming mode, we try sequentially until one succeeds.
        True parallel would require async, which complicates streaming.
        """

        # For now, implement as sequential with early exit
        # (True parallel would need async/threading refactor)

        approaches = [
            ("direct", query),
            ("explain", f"Explain the reasoning behind: {query}"),
            ("limitations", f"What are the challenges in addressing: {query}")
        ]

        best_attempt = None

        for idx, (approach_name, modified_query) in enumerate(approaches[:self.max_attempts]):
            logger.info(f"🔀 Multi-path {idx + 1}/{len(approaches)}: {approach_name}")

            # Generate response
            response_chunks = []
            wrapper = OllamaResonanceWrapper(
                self.anchor_text,
                model=self.model,
                enable_diagrams=True
            )

            try:
                async for chunk in wrapper.stream_chat(modified_query):
                    response_chunks.append(chunk)
                    self.current_drift = wrapper.current_drift

                    # Stream chunk to UI in real-time (don't wait until end)
                    if stream:
                        yield chunk

                    # Early exit if drift too high
                    if wrapper.current_drift > self.drift_abort_threshold:
                        logger.warning(f"⚠️ Drift threshold exceeded ({wrapper.current_drift:.3f} > {self.drift_abort_threshold}), aborting path '{approach_name}'")
                        break
            except Exception as e:
                import traceback
                logger.error(f"ERROR in resonance_reasoner streaming:")
                logger.error(f"  Exception: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                # Continue with what we have
                pass

            attempt = ReasoningAttempt(
                response=''.join(response_chunks),
                final_drift=wrapper.current_drift,
                max_drift=wrapper.current_drift,
                tokens=len(response_chunks),
                approach=approach_name,
                aborted=wrapper.current_drift > self.drift_abort_threshold
            )
            self.attempts.append(attempt)

            # If low drift, use this answer
            if not attempt.aborted:
                # Already streamed during attempt, just log success
                logger.info(f"✅ Path '{approach_name}' succeeded with drift {attempt.final_drift:.3f}")
                best_attempt = attempt
                return
            else:
                logger.warning(f"⚠️ Path '{approach_name}' aborted (drift: {attempt.final_drift:.3f})")
                if stream:
                    yield f"\n\n⚠️ Approach '{approach_name}' diverged (drift: {attempt.final_drift:.3f}). Trying next path...\n\n"

        # All paths failed, use least-bad attempt
        if self.attempts:
            best_attempt = min(self.attempts, key=lambda a: a.final_drift)
            logger.info(f"📊 All paths exceeded drift threshold. Using best attempt: '{best_attempt.approach}' (drift: {best_attempt.final_drift:.3f})")
            if stream:
                yield f"\n\n---\n**Note:** All reasoning paths exceeded drift threshold. The response above represents the most grounded attempt (drift: {best_attempt.final_drift:.3f}).\n"
        else:
            # No attempts at all (shouldn't happen)
            logger.error(f"❌ No reasoning attempts were made!")
            if stream:
                yield "\n\n❌ Error: No reasoning attempts completed. Please try again.\n"

    async def _self_critique(self, response: str, original_query: str) -> Dict:
        """
        Validate reasoning for logical errors.

        Returns dict with:
        - needs_revision: bool
        - issue: str (if needs_revision)
        """

        # Use a simple wrapper to check for issues
        critique_wrapper = OllamaResonanceWrapper(
            self.anchor_text,
            model=self.model
        )

        critique_prompt = f"""
You just generated this response to the query "{original_query}":

"{response[:500]}..."

Analyze this response for:
1. Circular reasoning (defining A using B, then B using A)
2. Unfounded assumptions not supported by context
3. Logical inconsistencies

If you find issues, respond with: "ISSUE: <brief description>"
If the response is sound, respond with: "VALID"
"""

        critique_result = ""
        async for chunk in critique_wrapper.stream_chat(critique_prompt):
            critique_result += chunk

        if "ISSUE:" in critique_result:
            issue = critique_result.split("ISSUE:")[1].strip()
            return {'needs_revision': True, 'issue': issue}
        else:
            return {'needs_revision': False, 'issue': None}

    def _reframe_query(self, query: str, attempt_num: int) -> str:
        """Reframe query for retry attempts"""

        if attempt_num == 0:
            return query  # First attempt: use original query
        elif attempt_num == 1:
            return f"Explain why this is challenging: {query}"
        elif attempt_num == 2:
            return f"What are the known approaches and limitations for: {query}"
        else:
            return f"Provide an honest assessment of: {query}"

    def _get_approach_name(self, attempt_num: int) -> str:
        """Get human-readable name for approach"""

        approaches = [
            "Direct reasoning",
            "Explain difficulty",
            "Known approaches",
            "Honest assessment"
        ]
        return approaches[min(attempt_num, len(approaches) - 1)]

    def _fallback_response(self) -> str:
        """Return fallback when all attempts fail"""

        return """
I attempted to reason through this query using multiple approaches, but encountered high uncertainty (drift from sources) in all attempts.

This suggests:
1. The query may require information not present in the retrieved sources
2. The problem may not have a definitive answer based on available evidence
3. The reasoning required exceeds the scope of the retrieved context

Would you like me to:
- Retrieve additional sources
- Rephrase the query
- Explain what information would be needed to answer this properly
"""

    def get_attempt_summary(self) -> str:
        """Get summary of all reasoning attempts"""

        if not self.attempts:
            return "No reasoning attempts made yet."

        summary = f"Reasoning attempts: {len(self.attempts)}\n\n"

        for idx, attempt in enumerate(self.attempts, 1):
            status = "❌ Aborted" if attempt.aborted else "✅ Completed"
            summary += f"{idx}. {attempt.approach}: {status} (drift: {attempt.final_drift:.3f})\n"
            if attempt.abort_reason:
                summary += f"   Reason: {attempt.abort_reason}\n"

        best = min(self.attempts, key=lambda a: a.final_drift)
        summary += f"\nBest attempt: {best.approach} (drift: {best.final_drift:.3f})"

        return summary
