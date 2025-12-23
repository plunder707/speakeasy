#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_chainlit.py - Chainlit Web UI for Resonance Governor

Features:
- Browser-based chat interface
- Display actual images from LLaVA
- Stream LLM responses in real-time
- Conversation history
- Source tracking
"""

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import chainlit as cl
from pathlib import Path

# Backend components
from wiki_governor import (
    VectorKnowledgeBase,
    fetch_wikipedia_full,
    fetch_arxiv_full,
    fetch_web_full,
    understand_images_with_llava,
    get_search_info
)
from resonance_reasoner import ResonanceReasoner
from agentic_rag import AgenticRAG

# Global state
kb = VectorKnowledgeBase()
session_counter = 0


@cl.on_chat_start
async def start():
    """Initialize chat session"""
    await cl.Message(
        content="**Resonance Governor - Agentic Mode 🤖**\n\n"
        "**Autonomous Research System**\n"
        "Powered by: DeepSeek-R1 | Wikipedia | arXiv | Web | LLaVA Vision\n\n"
        "**How it works:**\n"
        "1. I autonomously decide what sources to search\n"
        "2. I evaluate my knowledge gaps after each search\n"
        "3. I stop searching when I have enough information\n"
        "4. I generate answers with drift detection and backtracking\n\n"
        "Ask me anything. I'll research autonomously and show you my decision-making process."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages with agentic research"""
    global session_counter
    session_counter += 1
    session_id = f"session_{session_counter}"

    query = message.content

    # Create status message with spinner
    status_msg = cl.Message(content="⏳ **Initializing autonomous research...**")
    await status_msg.send()

    # Track search rounds for UI feedback
    search_messages = []
    current_search_msg = None
    response_msg = None

    try:
        # Status callback for AgenticRAG
        async def status_callback(msg: str, is_stream=False):
            """Handle status updates from agentic RAG"""
            nonlocal status_msg, current_search_msg, response_msg

            # Handle streaming response (final answer generation)
            if is_stream:
                # Stream the chunk to UI in real-time
                if response_msg is None:
                    # Create response message on first chunk
                    response_msg = cl.Message(content="")
                    await response_msg.send()
                await response_msg.stream_token(msg)
                return

            # Update status for search rounds
            if msg.startswith("📚 Round") or msg.startswith("🧠 Round"):
                # New round started
                if current_search_msg:
                    await current_search_msg.update()
                current_search_msg = cl.Message(content=f"⏳ {msg}")
                await current_search_msg.send()
                search_messages.append(current_search_msg)
            elif msg.startswith("   ✓") or msg.startswith("   ⚠️") or msg.startswith("   Decision"):
                # Update current round message
                if current_search_msg:
                    current_search_msg.content += f"\n{msg}"
                    await current_search_msg.update()
            elif msg.startswith("🔍 Executing"):
                # New search execution
                if current_search_msg:
                    current_search_msg.content += f"\n{msg}"
                    await current_search_msg.update()
            elif msg.startswith("✅ LLM ready"):
                # Ready to answer - remove spinner from last search msg
                if current_search_msg:
                    current_search_msg.content = current_search_msg.content.replace("⏳ ", "✅ ")
                    current_search_msg.content += f"\n{msg}"
                    await current_search_msg.update()
            elif msg.startswith("🎯 Generating"):
                # Final answer generation starting
                status_msg.content = f"🧠 **{msg}**"
                await status_msg.update()

        # Create agentic RAG instance with streaming enabled
        rag = AgenticRAG(max_search_rounds=5, enable_streaming=True)

        # Run autonomous research with real-time streaming
        status_msg.content = "⏳ **Running autonomous research...**"
        await status_msg.update()

        response_text, metadata = await rag.run(
            user_query=query,
            session_id=session_id,
            status_callback=status_callback
        )

        # Streaming already happened via callback, just finalize the message
        if response_msg:
            await response_msg.update()

        # Remove status message
        await status_msg.remove()

        # Add autonomous research summary
        drift_emoji = "✅" if metadata['final_drift'] < 0.5 else "⚠️" if metadata['final_drift'] < 0.85 else "❌"
        drift_status = "LOW" if metadata['final_drift'] < 0.5 else "MEDIUM" if metadata['final_drift'] < 0.85 else "HIGH"

        # Calculate token count from response
        token_count = len(response_text.split())

        summary_content = f"**🤖 Autonomous Research Summary**\n\n"
        summary_content += f"**Decision Making:**\n"
        summary_content += f"- Total search rounds: {metadata['total_search_rounds']}\n"
        summary_content += f"- Documents retrieved: {metadata['total_documents']}\n"
        summary_content += f"- Context size: {metadata['final_context_size']:,} chars\n\n"

        if metadata['search_history']:
            summary_content += f"**Search Decisions:**\n"
            for search in metadata['search_history']:
                conf_str = f"{search.get('confidence', 0):.2f}" if 'confidence' in search else "N/A"
                summary_content += f"- Round {search['round']}: {search['type'].upper()}[\"{search['query']}\"] "
                summary_content += f"→ {search['results']} docs (confidence: {conf_str})\n"
        else:
            summary_content += f"**Search Decisions:**\n"
            summary_content += f"- LLM decided initial search was sufficient ✅\n"

        summary_content += f"\n**Final Generation:**\n"
        summary_content += f"- Drift: {drift_emoji} {metadata['final_drift']:.3f} ({drift_status})\n"
        summary_content += f"- Reasoning attempts: {metadata['reasoning_attempts']}\n"
        summary_content += f"- Tokens generated: ~{token_count}\n"

        summary_msg = cl.Message(content=summary_content)
        await summary_msg.send()

    except Exception as e:
        import traceback
        await status_msg.remove()
        error_detail = traceback.format_exc()
        await cl.Message(content=f"❌ **Error:** {str(e)}\n\n```\n{error_detail}\n```").send()


if __name__ == "__main__":
    # This won't be called when using `chainlit run`
    # But included for clarity
    pass
