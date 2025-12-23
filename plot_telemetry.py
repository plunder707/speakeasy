import json
import matplotlib
# Force "headless" backend (fixes the crash on servers)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Load Data
try:
    with open("telemetry.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Run 'resonance_governor.py' first to generate data!")
    exit()

steps = [d['step'] for d in data]
drifts = [d['drift'] for d in data]
actions = [d['action'] for d in data]

# Setup Plot
plt.figure(figsize=(14, 7))
plt.style.use('dark_background')

# Zones
plt.axhspan(0, 0.50, color='#00ff00', alpha=0.1, label='Phase-Locked')
plt.axhspan(0.50, 0.85, color='#ffa500', alpha=0.15, label='Nudge Zone')
plt.axhspan(0.85, 1.2, color='#ff0000', alpha=0.2, label='Reject Zone')

# Plot Line
plt.plot(steps, drifts, color='cyan', linewidth=2, marker='o', label='Llama 3 Drift')

# Mark Rejections
for i, act in enumerate(actions):
    if act == "REJECT":
        plt.axvline(x=steps[i], color='red', linestyle='--', alpha=0.9)
        plt.text(steps[i], 0.9, ' REJECT', color='red', fontweight='bold', rotation=90)
    elif act == "NUDGE":
        plt.plot(steps[i], drifts[i], 'o', color='orange')

plt.title("Real-Time Telemetry: Resonance Governor vs Llama 3", fontsize=16, color='white')
plt.ylabel("Drift Score (1.0 = Chaos)", fontsize=12)
plt.xlabel("Cognitive Steps (Thoughts)", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

# SAVE the image
filename = "governor_telemetry.png"
plt.savefig(filename)
print(f"✅ Plot saved to '{filename}'")