# Architecture notes

## Pipeline stages

| Stage | Component | Output |
|---|---|---|
| Diarisation | transcript time-marks (swappable for VAD/pyannote) | patient-only audio, 16 kHz mono |
| Segmentation | utterance split | u₁ … u_U |
| Acoustic branch | WavLM-base-plus (frozen) + 8-d Praat prosody | 776-d token per utterance |
| Linguistic branch | Whisper large-v3-turbo → RoBERTa-base (frozen) | 768-d token per utterance |
| Fusion | bidirectional cross-attention + learned gate (d=128, 4 heads) | fused sequence H |
| Discourse | BiGRU (hidden 64) + attention pooling | discourse vector h |
| Bottleneck | Linear(128 → 24), MSE-supervised against measured markers | concepts c |
| Head | Linear(24 → 1) | logit + exact additive scorecard |
| Early-exit | max(p, 1−p) ≥ τ | stop / acquire more speech |

## What is trained vs. frozen

**Frozen (pretrained, never fine-tuned):** WavLM, RoBERTa, Whisper.
Rationale: 552 recordings cannot fine-tune encoders of this size without overfitting.

**Trained jointly:** cross-attention fusion → gate → BiGRU → attention pooling →
concept layer → linear head.

Gradients flow **through** the 24 concepts into the classifier, so the concept layer is
shaped by *both* losses at once: reproduce the real markers (MSE) **and** support the
correct diagnosis (BCE).

## Training objective

```
L = BCE(p, y) + λ · MSE(c, m),  λ = 5
```
`m` = the truly measured marker values for that recording.

## Hyper-parameters

Adam, lr 1e-3, weight decay 1e-4, dropout 0.3, batch 16, ≤80 epochs with early stopping
(patience 15) on validation AUC. Decision threshold tuned on a validation split (Youden's J).

## Design decisions worth knowing

| Decision | Why |
|---|---|
| WavLM + RoBERTa (over wav2vec2 + BERT) | AUC 0.710 → 0.722, stability ±0.04 → ±0.01, fidelity → 0.69 |
| Acoustic filled-pause detection | Whisper deletes "um/uh" from text; RMS of voiced gaps is ASR-independent |
| λ = 5 (not 8) and d = 128 (not 64) | λ=8 + smaller model dropped AUC 0.843 → 0.759 — reverted |
| Prosody concatenated to WavLM | mean-pooling washes out pitch; restored F0/jitter fidelity |
| Validation-tuned threshold | fixed 0.5 gave poor sensitivity/F1 at unchanged AUC |
| Repeated group k-fold | single splits were noisy (±0.04); repeated grouped CV gives ±0.014 |
