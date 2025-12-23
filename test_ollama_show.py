#!/usr/bin/env python3
"""Test ollama.show() response format"""

import ollama
import json

print("Testing ollama.show() format...\n")

try:
    info = ollama.show("deepseek-r1:8b")
    print(f"Type: {type(info)}")
    print(f"\nFull response:")
    print(json.dumps(info, indent=2, default=str))

    if hasattr(info, '__dict__'):
        print(f"\nObject attributes: {dir(info)}")

    # Check parameters field
    print(f"\n\nParameters field:")
    if isinstance(info, dict):
        params = info.get("parameters", {})
        print(f"  Type: {type(params)}")
        print(f"  Value: {params}")
    else:
        params = getattr(info, 'parameters', None)
        print(f"  Type: {type(params)}")
        print(f"  Value: {params}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print(traceback.format_exc())
