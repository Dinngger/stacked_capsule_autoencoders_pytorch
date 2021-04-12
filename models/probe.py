
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationProbe(nn.Module):
    def __init__(self, n_classes, d_feat):
        super(ClassificationProbe, self).__init__()
        self.fc1 = nn.Linear(n_classes, d_feat)

    def forward(self, features, labels, labeled=None):
        """Classification probe with stopped gradient on features."""

        logits = self.fc1(features.detach())
        xe = F.cross_entropy(logits, labels, reduction='none')
        if labeled is not None:
            xe = xe * labeled.float()
        xe = torch.mean(xe)
        acc = torch.mean(torch.eq(torch.argmax(logits, axis=1), labels).float())
        return xe, acc
