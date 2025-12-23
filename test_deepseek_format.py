#!/usr/bin/env python3
"""
Diagnostic script to examine DeepSeek-R1 response format
"""

import ollama
import json

print("Testing DeepSeek-R1 response format...\n")

# Simple test query
messages = [
    {"role": "user", "content": "What is 2+2? Explain your reasoning."}
]

try:
    stream = ollama.chat(
        model="deepseek-r1:8b",
        messages=messages,
        stream=True
    )

    chunk_count = 0
    for chunk in stream:
        chunk_count += 1

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_count}")
        print(f"{'='*60}")
        print(f"Type: {type(chunk)}")
        print(f"Full chunk: {json.dumps(chunk, indent=2, default=str)}")

        # Try to access message
        if isinstance(chunk, dict):
            if "message" in chunk:
                msg = chunk["message"]
                print(f"\nMessage type: {type(msg)}")
                print(f"Message keys: {msg.keys() if isinstance(msg, dict) else 'NOT A DICT'}")

                if isinstance(msg, dict):
                    print(f"  - thinking: {repr(msg.get('thinking', 'MISSING'))[:100]}")
                    print(f"  - content: {repr(msg.get('content', 'MISSING'))[:100]}")

        # Only show first 5 chunks
        if chunk_count >= 5:
            print("\n[Stopping after 5 chunks...]")
            break

except Exception as e:
    import traceback
    print(f"\nERROR: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")

print("\n✓ Test complete")
