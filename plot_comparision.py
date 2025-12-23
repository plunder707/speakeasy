import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load Data
try:
    with open("telemetry.json", "r") as f:
        gov_data = json.load(f)
    with open("baseline_telemetry.json", "r") as f:
        base_data = json.load(f)
except FileNotFoundError:
    print("❌ Missing data! Run 'resonance_governor.py' AND 'run_baseline.py' first.")
    exit()

# Prepare Lists
gov_steps = [d['step'] for d in gov_data]
gov_drifts = [d['drift'] for d in gov_data]
gov_actions = [d['action'] for d in gov_data]

base_steps = [d['step'] for d in base_data]
base_drifts = [d['drift'] for d in base_data]

# Plot
plt.figure(figsize=(14, 7))
plt.style.use('dark_background')

# Zones
plt.axhspan(0, 0.50, color='#00ff00', alpha=0.1, label='Phase-Locked')
plt.axhspan(0.50, 0.85, color='#ffa500', alpha=0.15, label='Nudge Zone')
plt.axhspan(0.85, 1.2, color='#ff0000', alpha=0.2, label='Reject Zone')

# 1. Baseline Line (The "Bad" Line)
plt.plot(base_steps, base_drifts, color='magenta', linewidth=2, linestyle=':', marker='x', label='Baseline (No Governor)')

# 2. Governor Line (The "Good" Line)
plt.plot(gov_steps, gov_drifts, color='cyan', linewidth=3, marker='o', label='With Resonance Governor')

# 3. Mark Rejections
for i, act in enumerate(gov_actions):
    if act == "REJECT":
        plt.axvline(x=gov_steps[i], color='red', linestyle='--', alpha=0.8)
        plt.text(gov_steps[i], 0.95, 'INTERVENTION', color='red', fontweight='bold', rotation=90)

plt.title("Impact Analysis: Resonance Governor vs Uncontrolled AI", fontsize=16, color='white')
plt.ylabel("Drift Score (1.0 = Hallucination)", fontsize=12)
plt.xlabel("Cognitive Steps", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("comparison_telemetry.png")
print("✅ Comparison saved to 'comparison_telemetry.png'")