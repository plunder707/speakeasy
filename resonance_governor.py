#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_governor.py
The Resonance Governor with telemetry and auto-model configuration.

Features:
- Real-time drift detection and correction
- Telemetry export to telemetry.json
- Automatic detection of model context length and optimal settings
"""

import sys
import logging
import re
import json
import time
from contextlib import aclosing
from typing import Generator, List, Dict, AsyncGenerator

try:
    from cowtest2 import ResonanceController, ControllerConfig, ActionType
except ImportError:
    print("Critical error: 'cowtest2.py' not found in the current directory.")
    sys.exit(1)

try:
    import ollama
    from ollama import AsyncClient
except ImportError:
    print("Critical error: 'ollama' library not found. Install with 'pip install ollama'.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("ResonanceGovernor")

# Global telemetry storage
telemetry_data = []

# Helper regex and constants
SENT_END = re.compile(r'([.!?])(\s|$)')
MIN_CHARS = 150


def looks_like_header(s: str) -> bool:
    s2 = s.strip()
    return len(s2) < 80 and (
        s2.startswith("#") or
        s2.endswith(":") or
        s2.isupper() or
        "Report" in s2 or
        "Overview" in s2
    )


def is_boilerplate(text: str) -> bool:
    t = text.lower()
    triggers = ["i apologize", "sorry", "revised report", "correction", "let me try again"]
    return any(trig in t for trig in triggers)


def detect_target_length(user_prompt: str) -> int:
    """
    Analyzes user prompt to determine appropriate response length target.
    Returns minimum character count before allowing completion.
    """
    lower = user_prompt.lower()

    # Very short answers
    if any(word in lower for word in ["minimal", "simple", "brief", "quick", "short"]):
        return 600

    # Deep research with visualization (explain + derive + create)
    # Check this BEFORE simple code examples
    if ("explain" in lower or "derive" in lower) and ("create" in lower or "visualiz" in lower):
        return 5000

    # Essays and comparisons
    if any(word in lower for word in ["essay", "compare", "contrast", "discuss", "evaluate"]):
        return 8000

    # Detailed analysis
    if any(word in lower for word in ["detail", "comprehensive", "thorough", "in-depth", "analyze"]):
        return 5000

    # Standard explanations
    if any(word in lower for word in ["explain", "describe", "what is", "how does"]):
        return 2000

    # Code examples (typically concise) - checked AFTER research queries
    if any(word in lower for word in ["write a", "function", "class", "code snippet"]):
        return 800

    # Default: moderate length
    return 1500


def has_natural_conclusion(text: str) -> bool:
    """
    Detects if response has reached a natural conclusion.
    """
    lower = text.lower()
    last_500 = lower[-500:] if len(lower) > 500 else lower

    conclusion_markers = [
        "in conclusion",
        "in summary",
        "to summarize",
        "therefore,",
        "thus,",
        "overall,",
        "i hope this helps",
        "let me know if you have",
        "feel free to ask"
    ]

    # Check for conclusion phrases
    if any(marker in last_500 for marker in conclusion_markers):
        return True

    # Check for code block completion (code + explanation)
    if "```" in text:
        # Count code blocks (must be even - all blocks closed)
        backtick_count = text.count("```")

        # If odd number of backticks, code block is still open - NOT complete
        if backtick_count % 2 != 0:
            return False

        # If we have complete code block(s) and explanation after, likely complete
        code_blocks = backtick_count // 2
        if code_blocks >= 1 and len(text.split("```")[-1].strip()) > 200:
            return True

    return False


def detect_repetition(recent_chunks: list, threshold: float = 0.85) -> bool:
    """
    Detects if recent chunks are too similar (repetition loop).
    Uses simple character-level similarity for speed.
    """
    if len(recent_chunks) < 3:
        return False

    # Check last 3 chunks
    last_three = recent_chunks[-3:]

    # Simple character overlap similarity
    for i in range(len(last_three) - 1):
        chunk1 = last_three[i].lower()
        chunk2 = last_three[i + 1].lower()

        # Calculate simple overlap ratio
        if len(chunk1) > 50 and len(chunk2) > 50:
            # Check if chunks are very similar
            overlap = sum(1 for c in chunk2 if c in chunk1) / len(chunk2)
            if overlap > threshold:
                return True

    return False


def trim_overlap(new_text: str, history_text: str) -> str:
    if not history_text or not new_text:
        return new_text
    check_len = min(len(history_text), 500)
    history_slice = history_text[-check_len:]
    max_overlap = 0
    for i in range(1, min(len(new_text), len(history_slice)) + 1):
        if history_slice.endswith(new_text[:i]):
            max_overlap = i
    return new_text[max_overlap:]


def should_evaluate(buf: str) -> bool:
    b = buf.strip()
    return len(b) >= MIN_CHARS and bool(SENT_END.search(b))


# --- Auto Model Configuration ---
def get_model_info(model_name: str) -> dict:
    try:
        return ollama.show(model_name)
    except Exception as e:
        logger.warning(f"Could not fetch model info for '{model_name}': {e}")
        return {}


def auto_config_for_model(model_name: str) -> dict:
    info = get_model_info(model_name)
    config = {
        "num_ctx": 8192,
        "num_predict": -1,      # unlimited output tokens
        "temperature": 0.7,
    }
    if info:
        # Extract context length from modelinfo (for DeepSeek-R1, Llama3, etc.)
        if hasattr(info, 'modelinfo'):
            modelinfo = info.modelinfo
            # Check for various context_length keys
            for key in ['qwen3.context_length', 'llama.context_length', 'context_length']:
                if key in modelinfo:
                    config["num_ctx"] = int(modelinfo[key])
                    break

        # Parse parameters string (DeepSeek-R1 format: "temperature    0.6\ntop_p    0.95")
        if hasattr(info, 'parameters') and isinstance(info.parameters, str):
            for line in info.parameters.split('\n'):
                if 'temperature' in line:
                    try:
                        # Extract value after "temperature"
                        parts = line.split()
                        if len(parts) >= 2:
                            config["temperature"] = float(parts[-1])
                    except (ValueError, IndexError):
                        pass

    logger.info(f"Auto-config for {model_name}: num_ctx={config['num_ctx']}, num_predict={config['num_predict']}, temperature={config['temperature']}")
    return config


class OllamaResonanceWrapper:
    def __init__(self, anchor_text: str, model: str = "llama3", device: str = "cuda", enable_diagrams: bool = False):
        config = ControllerConfig(
            drift_warn=0.50,
            drift_reject=0.85,
            evidence_threshold=0.15,
            novelty_threshold=0.40,
            base_temp=0.7,
            reject_temp=0.15
        )

        self.controller = ResonanceController(config, device=device)
        self.controller.set_ground_truth(anchor_text)
        self.anchor_text = anchor_text
        self.model = model
        self.enable_diagrams = enable_diagrams
        self.model_config = auto_config_for_model(model)  # Auto-detect settings
        self.history: List[Dict] = []
        self.current_drift = 0.0  # Real-time drift tracking for TUI

    async def stream_chat(self, user_prompt: str) -> AsyncGenerator[str, None]:
        logger.info(f"Starting generation with {self.model}")

        # Detect target length based on question type
        target_length = detect_target_length(user_prompt)
        logger.info(f"Target response length: {target_length} chars (detected from prompt)")

        # Base system instruction - Natural, flexible, alive
        base_instruction = f"""
        You are a thoughtful, capable AI assistant with deep reasoning abilities. Think naturally and independently.

        CRITICAL - Use your Chain-of-Thought reasoning:
        - Think step-by-step through problems like a human would
        - Show your genuine reasoning process - uncertainties, considerations, connections
        - Be curious and exploratory in your thinking
        - Question assumptions and consider multiple angles

        For this response, you have access to retrieved context/sources. Use them wisely:
        - Ground your answers in the provided sources when relevant
        - But also use your reasoning and knowledge to synthesize insights
        - If sources don't fully cover something, acknowledge it naturally: "The sources mention X, but don't cover Y..."
        - Don't force everything into a rigid research format

        Be natural and flexible:
        - If asked to write code, write excellent code with explanations
        - If asked a question, answer thoughtfully with reasoning
        - If having a conversation, be engaging and natural
        - Show personality - be curious, insightful, even playful when appropriate

        Think like deep critical thinker - independent, thoughtful, with soul. Not like a template-following robot.
        """

        # Add diagram generation instructions if enabled
        if self.enable_diagrams:
            visualization_instruction = """
        - After your text explanation, GENERATE Python visualization code to illustrate the concept if appropriate.

        Choose the appropriate visualization type:
        These are suggestions, but you may choose the most appropriate one based on the context:
        **For CYCLES/PATHWAYS** (Calvin cycle, Krebs cycle, metabolic pathways):
          → Use networkx with circular_layout

        **For GRAPHS/NETWORKS** (neural networks, protein interactions):
          → Use networkx with spring_layout or hierarchical_layout

        **For PLOTS/CHARTS** (data trends, functions, equations):
          → Use matplotlib (plt.plot, plt.scatter, plt.bar, etc.)

        **For FRACTALS/HEATMAPS** (Mandelbrot set, Julia sets, simulations):
          → Use numpy + matplotlib.pyplot.imshow() with colormaps

        **For 3D VISUALIZATIONS** (molecular structures, trajectories):
          → Use matplotlib's mplot3d or mayavi

        - Place code in a complete, runnable Python block with ALL imports
        - Use clear variable names and add comments
        - Include plt.show() or plt.savefig() at the end
            """
            base_instruction += visualization_instruction

        system_instruction = base_instruction + f"""

        --- CONTEXT START ---
        {self.anchor_text}
        --- CONTEXT END ---
        """

        self.history = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        current_temp = 0.7
        injection = None
        accepted_text = ""
        retries = 0
        # Reasoning models (DeepSeek-R1, o1) don't respond well to correction injections
        # They restart their thought process instead of continuing
        # For reasoning models, disable retries - let multi-path reasoner handle it
        is_reasoning_model = 'deepseek' in self.model.lower() or 'r1' in self.model.lower()
        MAX_RETRIES = 0 if is_reasoning_model else 3  # Reduced from 5 to 3 for regular models
        step_count = 0
        accepted_chunks = []  # Track chunks for repetition detection

        while True:
            request_history = list(self.history)
            if injection:
                request_history.append({"role": "user", "content": injection})

            try:
                logger.info(f"Calling AsyncClient.chat with model={self.model}")
                client = AsyncClient()
                stream = await client.chat(
                    model=self.model,
                    messages=request_history,
                    stream=True,
                    options={
                        "temperature": current_temp,
                        "num_ctx": self.model_config["num_ctx"],
                        "num_predict": self.model_config["num_predict"],
                    }
                )
                logger.info(f"AsyncClient.chat returned successfully, stream type: {type(stream)}")
            except Exception as e:
                import traceback
                logger.error(f"Ollama error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return

            token_buffer = ""
            stream_finished_naturally = False
            chunk_count = 0

            logger.info(f"Starting stream from {self.model}...")

            async with aclosing(stream) as stream_ctx:
                async for chunk in stream_ctx:
                    chunk_count += 1

                    # Handle different response formats (llama3 vs deepseek-r1)
                    try:
                        # Ollama returns ChatResponse objects with .message attribute
                        if hasattr(chunk, 'message'):
                            msg = chunk.message

                            # Manual Tag Detection & Routing ---
                            raw_content = getattr(msg, 'content', '') or ''

                            # Check for tags in the raw content stream
                            if "<think>" in raw_content:
                                self._in_thinking_block = True
                                raw_content = raw_content.replace("<think>", "")

                            if "</think>" in raw_content:
                                self._in_thinking_block = False
                                raw_content = raw_content.replace("</think>", "")

                            # Route content based on state so existing formatting logic works
                            if hasattr(self, '_in_thinking_block') and self._in_thinking_block:
                                thinking_text = raw_content
                                content_text = ""
                            else:
                                thinking_text = ""
                                content_text = raw_content

                            # Format thinking as indented/collapsible (blockquote style)
                            if thinking_text:
                                text = thinking_text
                                if not hasattr(self, '_in_thinking_block_header_sent'):
                                    self._in_thinking_block_header_sent = True
                                    text = "\n\n**💭 Thinking:**\n" + text
                                else:
                                    text = text

                            elif content_text:
                                 # Close thinking block if transitioning to content
                                if hasattr(self, '_in_thinking_block_header_sent') and self._in_thinking_block_header_sent:
                                    self._in_thinking_block_header_sent = False
                                    text = "\n\n**Answer:**\n" + content_text
                                else:
                                    text = content_text
                            else:
                                text = ''

                            if text:
                                token_buffer += text
                        else:
                            logger.warning(f"Unexpected chunk type: {type(chunk)}")
                            continue
                    except (KeyError, TypeError, AttributeError) as e:
                        import traceback
                        logger.error(f"ERROR in chunk processing:")
                        logger.error(f"  Exception: {e}")
                        logger.error(f"  Chunk type: {type(chunk)}")
                        logger.error(f"  Chunk value: {chunk}")
                        logger.error(f"  Traceback: {traceback.format_exc()}")
                        continue

                    clean_chunk = trim_overlap(token_buffer, accepted_text)
                    if not clean_chunk:
                        continue

                    if is_boilerplate(token_buffer):
                        token_buffer = ""
                        continue

                    if "\n" in token_buffer and looks_like_header(token_buffer):
                        to_yield = trim_overlap(token_buffer, accepted_text)
                        if to_yield:
                            yield to_yield
                            accepted_text += to_yield
                            self.history.append({"role": "assistant", "content": to_yield})
                        token_buffer = ""
                        continue

                    if not should_evaluate(token_buffer):
                        continue

                    # Skip drift evaluation for DeepSeek-R1 "thinking" blocks
                    # Thinking uses different vocabulary than sources (meta-reasoning vs content)
                    # Only evaluate drift on the actual "content" output
                    if hasattr(self, '_in_thinking_block') and self._in_thinking_block:
                        # Don't evaluate thinking - just yield it
                        to_yield = trim_overlap(token_buffer, accepted_text)
                        if to_yield:
                            yield to_yield
                            accepted_text += to_yield
                        token_buffer = ""
                        continue

                    step_count += 1
                    state, metrics = self.controller.evaluate_chunk(token_buffer)

                    # Update current drift for real-time TUI tracking
                    self.current_drift = float(metrics['drift'])

                    # Telemetry
                    telemetry_data.append({
                        "step": step_count,
                        "drift": float(metrics['drift']),
                        "action": state.action.name,
                        "chunk": token_buffer.strip()[:50] + "..."
                    })

                    if state.action == ActionType.REJECT:
                        logger.warning(f"REJECT: Drift={metrics['drift']:.2f}, Evidence={metrics['evidence']:.2f}")
                        retries += 1
                        if retries >= MAX_RETRIES:
                            if is_reasoning_model:
                                # For reasoning models, this is expected - multi-path will try different approach
                                logger.info(f"Path diverged from sources (drift={metrics['drift']:.2f}), trying alternative approach")
                                return  # Exit cleanly without error message
                            else:
                                # For regular models, this is a failure
                                yield "\n[System: Resonance Failure - Maximum correction attempts exceeded]\n"
                                return

                        current_temp = state.temperature
                        injection = "Correct the previous response to align with the context. Continue."
                        token_buffer = ""
                        break

                    # Accept
                    retries = 0
                    injection = None
                    current_temp = state.temperature

                    to_yield = trim_overlap(token_buffer, accepted_text)
                    if to_yield:
                        yield to_yield
                        accepted_text += to_yield
                        accepted_chunks.append(to_yield)  # Track for repetition
                        self.history.append({"role": "assistant", "content": to_yield})
                    token_buffer = ""

            # Stream has finished - mark it regardless of buffer content
            logger.info(f"Stream finished. Received {chunk_count} chunks total.")

            # The final flush logic below will handle any remaining buffer
            if not injection:
                stream_finished_naturally = True

            if stream_finished_naturally and token_buffer:
                state, metrics = self.controller.evaluate_chunk(token_buffer)
                # Update current drift for TUI
                self.current_drift = float(metrics['drift'])
                if state.action != ActionType.REJECT:
                    telemetry_data.append({
                        "step": step_count + 1,
                        "drift": float(metrics['drift']),
                        "action": state.action.name,
                        "chunk": "FINAL_FLUSH"
                    })
                    to_yield = trim_overlap(token_buffer, accepted_text)
                    if to_yield:
                        yield to_yield
                        accepted_text += to_yield

            if stream_finished_naturally:
                # Dynamic completion logic
                has_conclusion = has_natural_conclusion(accepted_text)
                is_repeating = detect_repetition(accepted_chunks)
                meets_target = len(accepted_text) >= target_length

                # Stop if: repeating OR (has conclusion AND meets minimum)
                if is_repeating:
                    logger.info(f"🔄 Repetition detected - stopping at {len(accepted_text)} chars")
                    break

                if has_conclusion and len(accepted_text) >= 2000:  # Increased minimum
                    logger.info(f"✓ Natural conclusion detected at {len(accepted_text)} chars")
                    break

                # Continue if below target and no conclusion
                if not meets_target:
                    logger.info(f"📝 Continuing (current: {len(accepted_text)}, target: {target_length})")
                    self.history.append({"role": "user", "content": "Continue the response with additional relevant details."})
                    continue

                break

        # CLEANUP: Always runs after stream ends
        # Close thinking block if still open
        if hasattr(self, '_in_thinking_block') and self._in_thinking_block:
            self._in_thinking_block = False
            logger.info("🔒 Closed thinking block")

        # Calculate final drift from all accumulated text
        if accepted_text and len(accepted_text) > 100:
            _, final_metrics = self.controller.evaluate_chunk(accepted_text[-2000:])  # Last 2000 chars
            self.current_drift = float(final_metrics['drift'])
            logger.info(f"📊 Final drift: {self.current_drift:.3f} (calculated from {len(accepted_text)} chars)")

        logger.info(f"✅ Generation complete: {len(accepted_text)} chars, drift: {self.current_drift:.3f}")


if __name__ == "__main__":
    rag_context = """
    The Resonance Governor is a software closed-loop control system for LLMs based on the Kuramoto model.
    It treats the AI text generation as a chaotic oscillator and the Context as a Master Oscillator.
    The system uses two sensors: a Semantic Sensor (Cosine Similarity via Embeddings) and an Evidence Verifier (Keyword Overlap).
    Core Mechanism: Vector-Space Entrainment. The Semantic Sensor calculates a running average of the 'Thought Vector'.
    If the AI drifts (hallucinates), the Governor increases 'Coupling Strength' by lowering the temperature (Nudge) or rejecting the token stream (Intervene).
    Control Logic: It uses a Hysteresis Loop with three states: MAINTAIN (Green), NUDGE (Orange), and REJECT (Red).
    This prevents 'Alien Encryption Key' style hallucinations and ensures the output remains Phase-Locked to ground truth.
    It is purely software, running on NVIDIA hardware (e.g., 4090) alongside inference engines like Ollama.
    """

    MODEL = "llama3"

    try:
        wrapper = OllamaResonanceWrapper(rag_context, model=MODEL, device="cuda")
    except Exception as e:
        print(f"Initialisation failed: {e}")
        sys.exit(1)

    print(f"\nAnchor set. Controlling {MODEL}.\n")

    prompt = "Write a detailed technical report explaining the Resonance Governor."
    for chunk in wrapper.stream_chat(prompt):
        print(chunk, end="", flush=True)

    # Save telemetry
    with open("telemetry.json", "w") as f:
        json.dump(telemetry_data, f, indent=4)
    print("\n\nTelemetry saved to 'telemetry.json'")
