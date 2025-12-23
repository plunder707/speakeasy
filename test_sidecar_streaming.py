#!/usr/bin/env python3
"""
Test sidecar.py async streaming with Ollama
Validates: async generator → sidecar.stream() → sync consumption
"""

import asyncio
from sidecar import Sidecar

try:
    import ollama
except ImportError:
    print("❌ ollama not found")
    exit(1)


async def async_ollama_stream(prompt: str):
    """Async generator that yields tokens from Ollama"""
    print("🔄 Starting async Ollama stream...")

    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        options={'temperature': 0.7, 'num_predict': 200}
    )

    for chunk in response:
        token = chunk['message']['content']
        yield token
        await asyncio.sleep(0)  # Yield control


def test_sidecar_streaming():
    """Test sidecar bridging async ollama to sync consumption"""
    print("🧪 Testing sidecar.py async→sync streaming\n")

    bridge = Sidecar(workers=2, name="OllamaStreamTest")

    prompt = "Explain photosynthesis in one sentence."

    print(f"Query: {prompt}\n")
    print("Response: ", end="", flush=True)

    try:
        # This is the key test: async generator → sidecar.stream() → sync for loop
        token_count = 0
        for token in bridge.stream(async_ollama_stream(prompt)):
            print(token, end="", flush=True)
            token_count += 1

        print(f"\n\n✅ Success! Streamed {token_count} tokens via sidecar")
        return True

    except Exception as e:
        print(f"\n\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    success = test_sidecar_streaming()
    exit(0 if success else 1)
