# Results — full DementiaBank Pitt (552 recordings, 292 speakers)

Protocol: speaker-independent, repeated (×3) stratified **group** 5-fold cross-validation.
All numbers are out-of-fold over every recording. Zero speaker leakage (asserted in code).

## Model comparison

| Model | Type | OOF ROC-AUC | F1 |
|---|---|---|---|
| Logistic regression (24 markers) | Interpretable | 0.665 | 0.666 |
| Random forest (24 markers) | Semi-interpretable | 0.716 | 0.716 |
| Cross-attention + MLP head | Black box | 0.719 | 0.717 |
| **Concept bottleneck (ours)** | Interpretable | **0.722 ± 0.014** | 0.695 |

**Interpretability cost** (black box − bottleneck) = **−0.003 AUC**.

## Full metric suite (proposed model)

| Metric | Mean ± SD | Metric | Mean ± SD |
|---|---|---|---|
| ROC-AUC | 0.722 ± 0.014 | Sensitivity | 0.669 ± 0.048 |
| PR-AUC | 0.738 ± 0.009 | Specificity | 0.679 ± 0.062 |
| Accuracy | 0.673 ± 0.003 | Precision | 0.729 ± 0.025 |
| Balanced accuracy | 0.674 ± 0.008 | F1 | 0.695 ± 0.015 |
| MCC | 0.347 ± 0.014 | | |

## Concept fidelity (explanation faithfulness)

Mean **r = 0.69**; 17 of 24 markers above r = 0.6.
Strongest: shimmer 0.94, speech rate 0.91, MLU 0.87, HNR 0.87, F0 std 0.87, RMS 0.87, TTR 0.86.
Weakest: lexical density 0.32, long pause 0.40, filled pause 0.46 — harder to recover from
mean-pooled encoder representations.

## Adaptive early-exit

| Seconds of patient speech | 10 | 20 | 30 | 45 | 60 | 90 | full |
|---|---|---|---|---|---|---|---|
| OOF ROC-AUC | 0.687 | 0.696 | 0.701 | 0.701 | 0.705 | 0.705 | 0.705 |

Knee at **≈ 10 s** (within 0.02 AUC of the full recording).

## Cross-dataset reference

| Dataset | Setting | ROC-AUC |
|---|---|---|
| Full Pitt (552) | Speaker-independent, unbalanced, multi-session (hard) | 0.722 |
| ADReSSo-2021 (166) | Balanced challenge subset | ≈ 0.84 |
