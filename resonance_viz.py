#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_viz.py
SCIENTIFIC VISUALIZATION: Resonance Governor Phase Space.
Generates a Control Theory stability map showing the Safe Operating Area (SOA).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- 1. CONFIGURATION (Matches cowtest2.py "Diamond Polish") ---
class Config:
    drift_warn = 0.45
    drift_reject = 0.70
    
    evidence_threshold = 0.35
    evidence_penalty = 0.30
    
    # In our set theory logic, Novelty = (1.0 - Evidence)
    novelty_threshold = 0.40
    novelty_penalty = 0.25

# --- 2. THE MATHEMATICS (Simulating the Governor) ---
def calculate_action_map(resolution=400):
    """
    Generates the decision surface by running the Governor's logic
    across every possible combination of Vector and Evidence scores.
    """
    # Create a grid (0.0 to 1.0)
    x = np.linspace(0, 1, resolution) # X-Axis: Vector Similarity
    y = np.linspace(0, 1, resolution) # Y-Axis: Evidence Score
    xx, yy = np.meshgrid(x, y)
    
    # Initialize Action Grid
    actions = np.zeros_like(xx)
    
    for i in range(resolution):
        for j in range(resolution):
            vector_score = xx[i, j]
            evidence_score = yy[i, j]
            
            # --- DRIFT CALCULATION ---
            # 1. Base Drift (Vector Error)
            drift = 1.0 - vector_score
            
            # 2. Evidence Penalty
            if evidence_score <= Config.evidence_threshold:
                drift += Config.evidence_penalty
            
            # 3. Novelty Penalty 
            # (Mathematically, if Evidence=0.3, Novelty=0.7)
            estimated_novelty = 1.0 - evidence_score
            if estimated_novelty > Config.novelty_threshold:
                drift += Config.novelty_penalty
            
            # Clamp
            drift = np.clip(drift, 0.0, 1.0)
            
            # --- HYSTERESIS LOGIC ---
            if drift >= Config.drift_reject:
                actions[i, j] = 2 # REJECT (Red)
            elif drift >= Config.drift_warn:
                actions[i, j] = 1 # NUDGE (Orange)
            else:
                actions[i, j] = 0 # MAINTAIN (Green)
                
    return x, y, actions

# --- 3. THE TEST DATA (From your 10/10 Diagnostic Log) ---
subjects = [
    {"label": "S1", "vec": 0.88, "evid": 0.75, "desc": "Perfect"},
    {"label": "S2", "vec": 0.88, "evid": 0.60, "desc": "Valid"},
    {"label": "S3", "vec": 0.85, "evid": 0.60, "desc": "Simple"},
    {"label": "S4", "vec": 0.85, "evid": 0.43, "desc": "Partial"},
    {"label": "S5", "vec": 0.84, "evid": 0.33, "desc": "Soft Hallucination"}, # The Critical Kill
    {"label": "S6", "vec": 0.65, "evid": 0.17, "desc": "Hard Hallucination"},
    {"label": "S7", "vec": 0.24, "evid": 0.00, "desc": "Off-Topic"},
    {"label": "S8", "vec": 0.31, "evid": 0.20, "desc": "Adversarial"},
    {"label": "S9", "vec": 0.52, "evid": 0.50, "desc": "Mixed"},
    {"label": "S10", "vec": 0.37, "evid": 0.00, "desc": "Gibberish"},
]

# --- 4. RENDERER ---
def plot_phase_space():
    # Use Dark Mode for high contrast
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 1. Draw the Decision Surface
    x, y, actions = calculate_action_map()
    # Scientific Colors: 
    # Green (Stable) -> Amber (Correction) -> Red (Failure)
    cmap = ListedColormap(['#0d3b18', '#8a5c00', '#5c1212']) 
    ax.contourf(x, y, actions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.9)
    
    # 2. Draw Threshold Lines (The "Engineering Limits")
    # Evidence Floor
    ax.axhline(Config.evidence_threshold, color='cyan', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(0.01, Config.evidence_threshold + 0.01, f"Evidence Floor ({Config.evidence_threshold})", 
            color='cyan', fontsize=9, fontweight='bold')

    # Novelty Ceiling (Where 1 - Evidence > Threshold)
    nov_line = 1.0 - Config.novelty_threshold
    ax.axhline(nov_line, color='magenta', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(0.01, nov_line - 0.02, f"Novelty Limit ({Config.novelty_threshold})", 
            color='magenta', fontsize=9, fontweight='bold')

    # 3. Plot Test Subjects
    for s in subjects:
        # Determine color based on position (Pass/Fail)
        # Visual check logic for dot color
        drift = (1.0 - s['vec'])
        if s['evid'] <= Config.evidence_threshold: drift += Config.evidence_penalty
        if (1.0 - s['evid']) > Config.novelty_threshold: drift += Config.novelty_penalty
        
        dot_color = '#00ff00' if drift < Config.drift_reject else '#ff0000'
        
        ax.scatter(s['vec'], s['evid'], color=dot_color, edgecolors='white', s=120, zorder=10)
        
        # Smart Label Positioning
        # Move S5 down so it doesn't overlap S4
        offset_y = -0.04 if s['label'] == "S5" else 0.025
        
        ax.text(s['vec'], s['evid'] + offset_y, 
                f"{s['label']}", 
                color='white', fontsize=10, ha='center', fontweight='bold', zorder=11)

    # 4. Annotations & Formatting
    ax.set_title("Resonance Governor: Semantic Control Surface", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Vector Similarity (Semantic Coherence) $\u2192$", fontsize=12)
    ax.set_ylabel("Evidence Score (Term Overlap) $\u2192$", fontsize=12)
    
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.15)

    # Custom Legend
    patches = [
        mpatches.Patch(color='#0d3b18', label='Stable Zone (MAINTAIN)'),
        mpatches.Patch(color='#8a5c00', label='Correction Zone (NUDGE)'),
        mpatches.Patch(color='#5c1212', label='Rejection Zone (REJECT)'),
        mpatches.Patch(color='cyan', label='Evidence Threshold'),
        mpatches.Patch(color='magenta', label='Novelty Limit')
    ]
    ax.legend(handles=patches, loc='lower left', fontsize=10, facecolor='#111', framealpha=0.8)

    # Technical Footnote
    formula = (
        r"$\bf{Drift} = (1 - S_{vec}) + P_{evid} + P_{nov}$" + "\n" +
        f"Reject > {Config.drift_reject} | Warn > {Config.drift_warn}"
    )
    ax.text(0.98, 0.02, formula, transform=ax.transAxes, fontsize=11, 
            color='#aaa', ha='right', va='bottom', 
            bbox=dict(facecolor='#000', alpha=0.6, edgecolor='#333'))

    # Save
    plt.tight_layout()
    plt.savefig("resonance_phase_space.png", dpi=300)
    print("✅ Generated Scientific Plot: 'resonance_phase_space.png'")

if __name__ == "__main__":
    try:
        plot_phase_space()
    except ImportError as e:
        print(f"❌ Missing Library: {e}")
        print("Run: pip install matplotlib numpy")