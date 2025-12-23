#!/usr/bin/env python3
"""
Test the full reasoning pipeline with DeepSeek-R1
"""

from resonance_reasoner import ResonanceReasoner

# Simple test context
anchor_text = """
The Pythagorean theorem states that in a right triangle, the square of the
hypotenuse (the side opposite the right angle) equals the sum of the squares
of the other two sides. This can be written as: a² + b² = c²

This theorem has been known since ancient times and appears in Euclid's Elements.
It is fundamental to geometry and has countless applications in mathematics,
physics, engineering, and computer science.
"""

print("Testing DeepSeek-R1 reasoning pipeline...")
print("=" * 60)

# Create reasoner with backtracking
reasoner = ResonanceReasoner(
    anchor_text,
    model="deepseek-r1:8b",
    drift_abort_threshold=0.75,
    max_attempts=2,
    enable_multipath=True,
    enable_critique=False  # Disable for quick test
)

query = "What is the Pythagorean theorem and how is it written?"

print(f"\nQuery: {query}\n")
print("Response:")
print("-" * 60)

try:
    for chunk in reasoner.reason_with_backtracking(query, stream=True):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60)
    print(f"\n✓ Reasoning complete!")
    print(f"  Final drift: {reasoner.current_drift:.3f}")
    print(f"  Attempts made: {len(reasoner.attempts)}")

    if reasoner.attempts:
        for idx, attempt in enumerate(reasoner.attempts, 1):
            status = "❌ Aborted" if attempt.aborted else "✅ Success"
            print(f"  {idx}. {attempt.approach}: {status} (drift: {attempt.final_drift:.3f})")

except Exception as e:
    import traceback
    print(f"\n\n❌ ERROR: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
