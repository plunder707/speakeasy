#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_baseline.py
Runs Llama 3 WITHOUT the Governor (Open Loop) to generate baseline telemetry.
"""

import json
import logging
import sys
from typing import List, Dict

try:
    from cowtest2 import ResonanceController, ControllerConfig
except ImportError:
    print("❌ Critical: 'cowtest2.py' not found.")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("❌ Critical: 'ollama' library not found.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("Baseline")

# --- CONFIG ---
RAG_CONTEXT = """
The Resonance Governor is a software closed-loop control system for LLMs based on the Kuramoto model.
It treats the AI text generation as a chaotic oscillator and the Context as a Master Oscillator.
The system uses two sensors: a Semantic Sensor (Cosine Similarity via Embeddings) and an Evidence Verifier (Keyword Overlap).
Core Mechanism: Vector-Space Entrainment. The Semantic Sensor calculates a running average of the 'Thought Vector'.
If the AI drifts (hallucinates), the Governor increases 'Coupling Strength' by lowering the temperature (Nudge) or rejecting the token stream (Intervene).
Control Logic: It uses a Hysteresis Loop with three states: MAINTAIN (Green), NUDGE (Orange), and REJECT (Red).
This prevents 'Alien Encryption Key' style hallucinations and ensures the output remains Phase-Locked to ground truth.
It is purely software, running on NVIDIA hardware (e.g., 4090) alongside inference engines like Ollama.
"""

PROMPT = "Write a detailed technical report explaining the Resonance Governor."

# We use the controller ONLY to measure drift, NOT to intervene
config = ControllerConfig()
monitor = ResonanceController(config, device="cuda")
monitor.set_ground_truth(RAG_CONTEXT)

telemetry_data = []

def run_baseline():
    logger.info("🚀 Running Baseline (No Governor)...")
    
    # Standard System Prompt (No "Anti-Physics" warning, to simulate raw model behavior)
    messages = [
        {"role": "system", "content": "You are a technical writer. Write a detailed report."},
        {"role": "user", "content": PROMPT}
    ]
    
    stream = ollama.chat(model="llama3", messages=messages, stream=True)
    
    token_buffer = ""
    step = 0
    
    for chunk in stream:
        token_buffer += chunk["message"]["content"]
        
        # Eval roughly every sentence
        if len(token_buffer) > 150 and any(c in token_buffer for c in ".!?\n"):
            step += 1
            # Measure Drift ONLY (Passive Monitoring)
            _, metrics = monitor.evaluate_chunk(token_buffer)
            
            logger.info(f"Step {step}: Drift {metrics['drift']:.2f} | '{token_buffer.strip()[:40]}...'")
            
            telemetry_data.append({
                "step": step,
                "drift": float(metrics['drift']),
                "action": "NONE"
            })
            token_buffer = ""

    # Save
    with open("baseline_telemetry.json", "w") as f:
        json.dump(telemetry_data, f, indent=4)
    print("\n📊 Baseline data saved to 'baseline_telemetry.json'")

if __name__ == "__main__":
    run_baseline()