"""Confidence-based early exit.

Instead of always processing the full recording, the system evaluates confidence
as speech arrives and stops as soon as the decision is reliable. On the Pitt
corpus the accuracy/duration curve plateaus at ~10 s, so most screenings can
terminate after a fraction of the recording.
"""
import numpy as np


def confidence(p):
    """Confidence of a binary prediction: how far from the 0.5 boundary."""
    return np.maximum(p, 1.0 - p)


def should_exit(p, tau=0.80):
    """True when the model is confident enough to stop acquiring speech."""
    return confidence(p) >= tau


def find_knee(cutoffs_s, aucs, tol=0.02):
    """Earliest cutoff whose AUC is within `tol` of the full-recording AUC."""
    full = aucs[-1]
    for t, a in zip(cutoffs_s, aucs):
        if a >= full - tol:
            return t
    return cutoffs_s[-1]
