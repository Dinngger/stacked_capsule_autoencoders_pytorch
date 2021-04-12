import math
import torch
# import torch.nn as nn
import torch.nn.functional as F


class MixtureDistribution:
    """Mixture."""

    @staticmethod
    def mixing_log_prob(mixing_logits, loc):
        return mixing_logits - torch.logsumexp(mixing_logits, 1, keepdim=True)

    @staticmethod
    def log_prob(mixing_logits, loc, x, mixing_log_prob):
        lp = -((loc - x) ** 2) / 2 - math.log(math.sqrt(2 * math.pi))
        return torch.logsumexp(lp + mixing_log_prob, 1)

    @staticmethod
    def mean(mixing_logits, loc):
        return torch.sum(F.softmax(mixing_logits, 1) * loc, 1)

    @staticmethod
    def mode(mixing_logits, loc, mixing_log_prob):
        """Mode of the distribution.

        Args:
        straight_through_gradient: Boolean; if True, it uses the straight-through
            gradient estimator for the mode. Otherwise there is no gradient
            with respect to the mixing coefficients due to the `argmax` op.
        maximum: if True, attempt to return the highest-density mode.

        Returns:
        Mode.
        """
        mask = F.one_hot(torch.argmax(mixing_log_prob, dim=1),
                         mixing_log_prob.shape[1])
        mask = mask.permute(0, 3, 1, 2)
        return torch.sum(mask.contiguous() * loc, 1)
