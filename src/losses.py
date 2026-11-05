"""Joint objective: task loss + concept supervision.

    L = BCE(p, y) + lambda * MSE(c, m)

The MSE term is what makes each bottleneck neuron represent its *named* clinical
marker m, so the explanation is faithful and its fidelity can be measured.
"""
import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()


def joint_loss(logits, y, concepts=None, markers=None, lambda_concept=5.0):
    loss = bce(logits, y.float())
    if concepts is not None and markers is not None:
        loss = loss + lambda_concept * mse(concepts, markers)
    return loss
