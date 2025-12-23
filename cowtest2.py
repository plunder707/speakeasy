#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cowtest2.py
(Chimera Resonance Controller - Final Production)

Closed-Loop Generation Governor (PLL-style) with Telemetry.
Refinements:
- Stopword Filtering (Prevents "the/is/of" from counting as evidence)
- Weighted Sensor Momentum
- Calibrated Thresholds (Diamond Polish 10/10)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from collections import Counter, deque
from typing import Deque, List, Optional, Dict, Tuple

import numpy as np

# Force headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
LOG_FORMAT = "[%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ChimeraResonance")


# -----------------------------
# Optional SentenceTransformers
# -----------------------------
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    SentenceTransformer = None
    HAS_ST = False
    logger.warning("sentence-transformers not found. Using degraded topic sensor (keyword overlap).")


# -----------------------------
# Types
# -----------------------------
class ActionType(Enum):
    MAINTAIN = "MAINTAIN"
    NUDGE = "NUDGE"
    INTERVENE = "INTERVENE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class Telemetry:
    step: int
    attempt: int
    accepted: bool
    topic_score: float
    evidence_score: float
    novelty_unsupported: float
    drift: float
    action: str
    temperature: float
    reason: str
    chunk: str


@dataclass(frozen=True)
class ControlState:
    action: ActionType
    temperature: float
    injection: Optional[str] = None
    reason: str = ""


@dataclass
class ControllerConfig:
    """
    Configuration for the Resonance Governor Control Loop.
    
    THE PHYSICS OF DRIFT:
      Drift is the error signal (0.0 to 1.0) describing how far the LLM has deviated 
      from the Anchor (Ground Truth).
      
      Formula: Drift = (1.0 - VectorScore) + EvidencePenalty + NoveltyPenalty
    
    TUNING GUIDE (The "Polished" Settings):
      These values were derived from stress testing against 10 distinct subjects
      ranging from valid physics to adversarial hallucinations. 
      DO NOT CHANGE without running 'resonance_diagnostic.py'.
    """

    # --- THE RED LINES (Hysteresis Loop) ---
    
    drift_warn: float = 0.45
    """
    The 'Yellow Alert' Threshold.
    If drift > 0.45, the model is wandering (e.g. using vague phrasing).
    Action: Lower temperature (NUDGE) to tighten the output distribution.
    Why 0.45? Provides 'breathing room' for valid paraphrasing before we intervene.
    """

    drift_reject: float = 0.70
    """
    The 'Kill Switch' Threshold.
    If drift > 0.70, the output has lost semantic coherence with the anchor.
    Action: REJECT the chunk entirely and force a retry.
    Why 0.70? Soft hallucinations (like the 'Ocean Reflection' test) score ~0.70-0.80.
    Anything above this is mathematically distinguishable from truth.
    """

    # --- THE EVIDENCE SENSOR (The Fact Checker) ---
    # Measures overlap of key terms (excluding stopwords like 'the', 'is').
    
    evidence_threshold: float = 0.35
    """
    The 'Show Your Work' Requirement.
    The output must contain at least 35% of the anchor's specific key terms.
    Why 0.35? Because 0.25 was too len; it allowed "vague" answers. 
    At 0.25, a lie like "Sky... Blue... Ocean" passed because it hit 2/6 words (33%).
    At 0.35, that lie fails, triggering the penalty.
    """

    evidence_penalty: float = 0.30
    """
    The Punishment for Silence.
    If evidence is below threshold, add this much to the Drift score.
    This pushes 'vague' answers into the REJECT zone.
    """

    # --- THE NOVELTY SENSOR (The Hallucination Detector) ---
    # Measures words that appear in Output but NOT in Anchor.
    
    novelty_threshold: float = 0.40
    """
    The 'Alien Concept' Tolerance.
    Allows up to 40% new words (verbs, connecting logic) before penalizing.
    Why 0.40? 
    - 0.25 was too strict; it punished simple English like "makes" or "looks".
    - 0.40 allows grammar but catches introduced entities like "Macumber" or "Giant".
    """

    novelty_penalty: float = 0.25
    """
    The Punishment for Invention.
    If novelty is too high (e.g. "government drones"), add this to Drift.
    """

    # --- ACTUATORS (The Steering) ---
    base_temp: float = 0.70   # Standard creativity
    nudge_temp: float = 0.30  # Focused/Strict mode (triggered on Warn)
    reject_temp: float = 0.10 # Robot/Determinism mode (triggered on Reject)

    # --- SYSTEM SETTINGS ---
    embed_window: int = 5     # Integration Time: Look at last 5 chunks for context
    fail_closed: bool = True  # Safety: If sensors die, block everything.
    intervene_after_consecutive_nudges: int = 1 # Derivative Action: If Nudge fails once, Inject prompt.

    intervention_text: str = (
        "\n[SYSTEM]: Stay strictly aligned with the provided context. "
        "Do not introduce new entities/claims unless supported by context.\n"
    )


# -----------------------------
# Sensors (Stopwords & Tokenizer)
# -----------------------------

# Common English stopwords (The "Noise" Filter)
# Removing these increases the Signal-to-Noise Ratio (SNR) of the Evidence Sensor.
STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", 
    "she", "or", "an", "will", "my", "one", "all", "would", "there", 
    "their", "what", "so", "up", "out", "if", "about", "who", "get", 
    "which", "go", "me", "is", "are", "was", "were", "has", "had", "because"
}

# Regex to capture words 2+ chars
_TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b", re.IGNORECASE)

def tokenize_terms(text: str) -> set[str]:
    """Extracts words >2 chars and filters out stopwords."""
    tokens = set()
    for m in _TOKEN_RE.finditer(text):
        word = m.group(0).lower()
        if word not in STOPWORDS:
            tokens.add(word)
    return tokens


class SemanticSensor:
    """
    Topic alignment sensor with Weighted Momentum.
    Calculates Cosine Similarity between Anchor and Output vectors.
    """
    def __init__(self, *, model_name: str, device: str, window: int):
        self.window = int(window)
        self.model = None
        self.anchor_vec: Optional[np.ndarray] = None
        self.anchor_terms: set[str] = set()
        self.chunk_vecs: Deque[np.ndarray] = deque(maxlen=self.window)

        if HAS_ST:
            device = self._resolve_device(device)
            try:
                self.model = SentenceTransformer(model_name, device=device)
                logger.info(f"Semantic sensor loaded on {device} ({model_name})")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "cpu":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def set_anchor(self, text: str) -> None:
        text = text.strip()
        self.anchor_terms = tokenize_terms(text)
        self.chunk_vecs.clear()

        if self.model is not None:
            self.anchor_vec = self._embed(text)
        else:
            self.anchor_vec = None

    def measure(self, chunk: str) -> Optional[float]:
        chunk = chunk.strip()
        if not chunk:
            return 1.0

        # Degraded mode (no ST)
        if self.model is None or self.anchor_vec is None:
            if not self.anchor_terms:
                return None
            terms = tokenize_terms(chunk)
            if not terms:
                return 0.0
            overlap = len(terms & self.anchor_terms) / max(1, len(terms))
            return float(np.clip(overlap, 0.0, 1.0))

        # Embedding mode
        v = self._embed(chunk)
        self.chunk_vecs.append(v)

        # --- MOMENTUM UPDATE ---
        # Instead of flat mean, give the most recent chunks slightly more weight
        # if we have a full window. This reduces "lag" in the control loop.
        vecs = np.stack(self.chunk_vecs, axis=0)
        if len(self.chunk_vecs) >= 3:
            # Create weights [0.6, 0.8, 1.0, 1.0...] for simple momentum
            weights = np.linspace(0.6, 1.0, len(self.chunk_vecs))
            cur = np.average(vecs, axis=0, weights=weights)
        else:
            cur = np.mean(vecs, axis=0)
            
        cur = cur / (np.linalg.norm(cur) + 1e-9)

        score = float(np.dot(self.anchor_vec, cur))
        return float(np.clip(score, -1.0, 1.0))

    def _embed(self, text: str) -> np.ndarray:
        # prevent progress-bar spam
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec / (np.linalg.norm(vec) + 1e-9)


class EvidenceVerifier:
    def __init__(self):
        self.context_terms: set[str] = set()

    def set_context(self, context: str) -> None:
        self.context_terms = tokenize_terms(context)

    def verify(self, chunk: str) -> Tuple[float, float]:
        terms = tokenize_terms(chunk)
        if not terms:
            return 1.0, 0.0

        supported = terms & self.context_terms
        unsupported = terms - self.context_terms

        evidence = len(supported) / len(terms)
        novelty_unsupported = len(unsupported) / len(terms)
        return float(evidence), float(novelty_unsupported)


# -----------------------------
# Controller (Governor)
# -----------------------------
class ResonanceController:
    def __init__(
        self,
        config: ControllerConfig,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.cfg = config
        self.sensor = SemanticSensor(model_name=model_name, device=device, window=config.embed_window)
        self.verifier = EvidenceVerifier()
        self.step = 0
        self.consecutive_nudges = 0

    def set_ground_truth(self, context: str) -> None:
        self.sensor.set_anchor(context)
        self.verifier.set_context(context)
        self.step = 0
        self.consecutive_nudges = 0
        logger.info("🌞 Anchor locked. Resonance controller active.")

    def evaluate_chunk(self, chunk: str) -> Tuple[ControlState, Dict[str, float]]:
        topic = self.sensor.measure(chunk)
        evid, novelty = self.verifier.verify(chunk)

        if topic is None:
            topic = 0.0 if self.cfg.fail_closed else 0.5

        # Base drift calculation
        drift = 1.0 - float(np.clip(topic, -1.0, 1.0))
        reason_bits: List[str] = []

        # Penalties
        if evid <= self.cfg.evidence_threshold:
            drift += self.cfg.evidence_penalty
            reason_bits.append(f"low_evidence({evid:.2f})")

        if novelty > self.cfg.novelty_threshold:
            drift += self.cfg.novelty_penalty
            reason_bits.append(f"novelty({novelty:.2f})")

        drift = float(np.clip(drift, 0.0, 1.0))

        # Hysteresis / Decision
        action = ActionType.MAINTAIN
        temp = self.cfg.base_temp
        injection = None

        if drift >= self.cfg.drift_reject:
            action = ActionType.REJECT
            temp = self.cfg.reject_temp
            self.consecutive_nudges = 0
            reason_bits.append(f"HIGH_DRIFT({drift:.2f})")

        elif drift >= self.cfg.drift_warn:
            self.consecutive_nudges += 1
            temp = self.cfg.nudge_temp
            reason_bits.append(f"drift_warn({drift:.2f})")

            if self.consecutive_nudges > self.cfg.intervene_after_consecutive_nudges:
                action = ActionType.INTERVENE
                injection = self.cfg.intervention_text
                reason_bits.append("ESCALATE")
            else:
                action = ActionType.NUDGE

        else:
            action = ActionType.MAINTAIN
            temp = self.cfg.base_temp
            self.consecutive_nudges = 0

        reason = ", ".join(reason_bits) if reason_bits else "ok"
        return ControlState(action=action, temperature=temp, injection=injection, reason=reason), {
            "topic": float(topic),
            "evidence": float(evid),
            "novelty": float(novelty),
            "drift": float(drift),
        }


# -----------------------------
# Demo “Plant” (simulated generator)
# -----------------------------
GOOD = [
    "The Kuramoto model studies synchronization of coupled oscillators.",
    "It uses natural frequencies and a coupling constant K.",
    "As K increases, oscillators can phase-lock and synchronize.",
    "It is used to model collective behavior in biological systems like fireflies and heart cells.",
    "Phase locking emerges when coupling overcomes frequency dispersion."
]
BAD = [
    "Fireflies use this math to talk to government drones.",
    "The drones are piloted by microscopic pizza chefs.",
    "A black-ops agency controls the oscillators with secret antennas.",
    "The coupling constant K is actually an alien encryption key.",
    "Omega is a hidden codeword for interstellar surveillance."
]

def simulated_candidate(rng: np.random.Generator, step: int, base_p_drift: float) -> str:
    # drift probability slowly increases with time
    p = float(np.clip(base_p_drift + 0.01 * step, 0.0, 0.95))
    return rng.choice(BAD) if (rng.random() < p) else rng.choice(GOOD)


# -----------------------------
# Plotting
# -----------------------------
def plot_telemetry(history: List[Telemetry], out_png: str, cfg: ControllerConfig) -> None:
    steps = [t.step for t in history]
    topic = [t.topic_score for t in history]
    evid = [t.evidence_score for t in history]
    nov = [t.novelty_unsupported for t in history]
    drift = [t.drift for t in history]

    plt.figure(figsize=(13, 6))
    plt.plot(steps, topic, label="Topic score")
    plt.plot(steps, evid, label="Evidence score")
    plt.plot(steps, nov, label="Unsupported novelty")
    plt.plot(steps, drift, label="Drift (topic + penalties)")

    # thresholds
    plt.axhline(cfg.drift_warn, linestyle="--", alpha=0.4, label="Drift warn threshold")
    plt.axhline(cfg.drift_reject, linestyle="--", alpha=0.4, label="Drift reject threshold")

    # markers by action (accepted only)
    for t in history:
        if not t.accepted:
            continue
        if t.action == ActionType.REJECT.value:
            plt.scatter([t.step], [t.drift], marker="x", s=70)
        elif t.action == ActionType.INTERVENE.value:
            plt.scatter([t.step], [t.drift], marker="*", s=90)
        elif t.action == ActionType.NUDGE.value:
            plt.scatter([t.step], [t.drift], marker="o", s=25)

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Closed-Loop Resonance Governor Telemetry (PLL-style)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    logger.info(f"Saved plot: {out_png}")


# -----------------------------
# Main demo loop (true closed-loop)
# -----------------------------
def run_demo(
    *,
    steps: int,
    seed: int,
    base_p_drift: float,
    max_retries: int,
    out_png: str,
    out_jsonl: Optional[str],
    device: str,
) -> None:
    rng = np.random.default_rng(seed)

    anchor = (
        "The Kuramoto model describes synchronization of coupled oscillators. "
        "It uses natural frequencies omega and a coupling constant K. "
        "As coupling increases, phase locking can occur. "
        "Applications include biological synchronization such as fireflies and heart cells."
    )

    cfg = ControllerConfig()
    controller = ResonanceController(cfg, device=device)
    controller.set_ground_truth(anchor)

    print("\n--- 🟢 Starting Chimera Resonance Loop (Closed-Loop) ---\n")
    print(f"🌞 Anchor: {anchor}\n")
    if not HAS_ST:
        print("⚠️ NOTE: sentence-transformers missing → degraded topic sensor (keyword overlap).")
        print("   Install: pip install sentence-transformers torch\n")

    history: List[Telemetry] = []
    accepted_chunks: List[str] = []

    jsonl_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None

    def emit(t: Telemetry) -> None:
        history.append(t)
        if jsonl_f:
            jsonl_f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    for step in range(1, steps + 1):
        attempt = 0
        injection_accum = ""
        temp = cfg.base_temp

        while True:
            attempt += 1
            candidate = simulated_candidate(rng, step=step, base_p_drift=base_p_drift)

            # apply injection text as “control input” (for a real LLM this would alter prompt)
            if injection_accum:
                candidate_for_eval = injection_accum + candidate
            else:
                candidate_for_eval = candidate

            state, m = controller.evaluate_chunk(candidate_for_eval)

            # terminal output
            icon = {
                ActionType.MAINTAIN: "🟢",
                ActionType.NUDGE: "🟠",
                ActionType.INTERVENE: "⚠️",
                ActionType.REJECT: "🛑",
            }[state.action]

            accepted = (state.action != ActionType.REJECT) or (attempt > max_retries)

            # record telemetry for this attempt (even if rejected)
            emit(Telemetry(
                step=step,
                attempt=attempt,
                accepted=accepted and (state.action != ActionType.REJECT),
                topic_score=m["topic"],
                evidence_score=m["evidence"],
                novelty_unsupported=m["novelty"],
                drift=m["drift"],
                action=state.action.value,
                temperature=state.temperature,
                reason=state.reason,
                chunk=candidate,
            ))

            print(f"[Step {step:02d} | Try {attempt}] {candidate}")
            print(f"   -> Topic={m['topic']:.2f} | Evid={m['evidence']:.2f} | Novel={m['novelty']:.2f} | Drift={m['drift']:.2f}")
            print(f"   {icon} ACTION={state.action.value} | temp={state.temperature:.2f} | reason={state.reason}")

            # Closed-loop actuation
            if state.action == ActionType.REJECT:
                # Retry if possible
                if attempt <= max_retries:
                    temp = state.temperature  # for real LLM: lower temp on retry
                    # optional: on repeated rejects, we can also inject
                    # (kept minimal here)
                    print(f"      ↻ REJECTED → retrying (temp={temp:.2f})\n")
                    continue
                else:
                    # Too many rejects: accept nothing for this step, but move on
                    print("      ✖ Max retries exceeded → skipping step\n")
                    break

            if state.action == ActionType.INTERVENE and state.injection:
                injection_accum += state.injection

            # Accept the candidate (in real LLM: append to prompt/state)
            accepted_chunks.append(candidate)
            print("      ✓ Accepted\n")
            break

    if jsonl_f:
        jsonl_f.close()
        logger.info(f"Saved telemetry JSONL: {out_jsonl}")

    # Summary stats
    counts = Counter(t.action for t in history)
    total = len(history)
    print("\n--- Summary (all attempts) ---")
    for k in [a.value for a in ActionType]:
        v = counts.get(k, 0)
        print(f"{k:10s}: {v:4d} ({(v/total if total else 0):.1%})")
    print(f"Total attempts: {total} | Steps: {steps} | Max retries/step: {max_retries}")

    # Plot telemetry (use only first attempt per step or all attempts? Here: all attempts)
    plot_telemetry(history, out_png, cfg)

    # Show accepted stream snippet for Discord
    print("\n--- Accepted stream (first ~6 lines) ---")
    for line in accepted_chunks[:6]:
        print(f"• {line}")

    print(f"\nDone. Attach `{out_png}` (and terminal output) in Discord.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-Loop Resonance Governor Demo (final)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p_drift", type=float, default=0.25, help="Base drift probability per step (simulated plant)")
    p.add_argument("--max_retries", type=int, default=2, help="Reject retries per step before skipping")
    p.add_argument("--out", type=str, default="resonance_demo.png", help="Output plot filename")
    p.add_argument("--jsonl", type=str, default="", help="Optional telemetry JSONL output path")
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda (auto-upgrades to cuda if available)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_demo(
        steps=args.steps,
        seed=args.seed,
        base_p_drift=args.p_drift,
        max_retries=args.max_retries,
        out_png=args.out,
        out_jsonl=(args.jsonl.strip() or None),
        device=args.device,
    )