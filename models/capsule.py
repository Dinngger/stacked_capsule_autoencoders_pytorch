from models.utils import geometric_transform
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

import collections
from monty.collections import AttrDict

from models.utils import safe_log, safe_ce, index_select_nd, normalize, l2_loss

td = torch.distributions


class CapsuleLayer(nn.Module):
    def __init__(self, n_caps, d_caps, d_feat, n_votes, n_caps_params, n_hidden=128):
        super(CapsuleLayer, self).__init__()
        self._n_transform_params = 6

        self._n_caps = n_caps
        self._n_caps_dims = d_caps
        self._n_caps_params = n_caps_params
        self._n_votes = n_votes

        self.conv0 = nn.Conv1d(d_feat, n_hidden, 1, 1)
        self.conv1 = nn.Conv1d(n_hidden, n_caps_params, 1, 1)

        self.output_shapes = (
            [self._n_votes, self._n_transform_params],  # CPR_dynamic
            [1, self._n_transform_params],  # CCR
            [1],  # per-capsule presence
            [self._n_votes],  # per-vote-presence
            [self._n_votes],  # per-vote scale
        )
        self.splits = [np.prod(i).astype(np.int32) for i in self.output_shapes]
        n_outputs = sum(self.splits)

        # we don't use bias in the output layer in order to separate the static
        # and dynamic parts of the CPR
        self.caps_mlp = nn.Sequential(
            nn.Conv1d(self._n_caps_params + 1, n_hidden, 1, 1, bias=False),
            nn.Conv1d(n_hidden, n_outputs, 1, 1, bias=False)
        )

        res_bias = []
        for i in range(1, 5):
            res_bias.append(nn.Parameter(torch.Tensor(*([1, 1] + self.output_shapes[i]))))
            nn.init.xavier_uniform_(res_bias[i-1])
        self.res_bias = nn.ParameterList(res_bias)

        self.cpr_static = nn.Parameter(torch.Tensor(1, self._n_caps, self._n_votes, self._n_transform_params))
        nn.init.xavier_uniform_(self.cpr_static)

    def forward(self, features, device):
        batch_shape = list(features.shape[0:2])
        caps_params = self.conv0(features.transpose(1, 2)).transpose(1, 2)
        caps_params = self.conv1(caps_params.transpose(1, 2)).transpose(1, 2)
        caps_params = caps_params.reshape(batch_shape + [self._n_caps_params])

        caps_exist = torch.ones(batch_shape + [1], dtype=torch.float, device=device)
        caps_params = torch.cat([caps_params, caps_exist], -1)

        all_params = self.caps_mlp(caps_params.transpose(1, 2)).transpose(1, 2)
        all_params = torch.split(all_params, self.splits, -1)
        res = [i.reshape(batch_shape + s) for (i, s) in zip(all_params, self.output_shapes)]

        cpr_dynamic = res[0]
        res = [res[i] + self.res_bias[i-1] for i in range(1, 5)]
        ccr, pres_logit_per_caps, pres_logit_per_vote, scale_per_vote = res

        ccr = self.make_transform(ccr)
        cpr = self.make_transform(cpr_dynamic.contiguous() + self.cpr_static)

        ccr_per_vote = ccr.repeat(1, 1, self._n_votes, 1, 1)
        votes = torch.matmul(ccr_per_vote, cpr)
        pres_per_caps = torch.sigmoid(pres_logit_per_caps)

        pres_per_vote = pres_per_caps * torch.sigmoid(pres_logit_per_vote)
        scale_per_vote = F.softplus(scale_per_vote + .5) + 1e-2

        return AttrDict(
            vote=votes,
            scale=scale_per_vote,
            vote_presence=pres_per_vote,
            pres_logit_per_caps=pres_logit_per_caps,
            pres_logit_per_vote=pres_logit_per_vote,
            dynamic_weights_l2=l2_loss(cpr_dynamic) / features.size(0),
            raw_caps_features=features,
        )

    def make_transform(self, params):
        return geometric_transform(params, as_matrix=True)


class CapsuleLikelihood(nn.Module):
    """Capsule voting mechanism."""

    OutputTuple = collections.namedtuple('CapsuleLikelihoodTuple',
                                         ('log_prob vote_presence winner '
                                          'winner_pres is_from_capsule '
                                          'mixing_logits mixing_log_prob '
                                          'soft_winner soft_winner_pres '
                                          'posterior_mixing_probs'))

    def __init__(self, votes_shape):
        super(CapsuleLikelihood, self).__init__()
        self._n_votes = 1
        self._n_caps = int(votes_shape[1])
        self.dummy_vote = nn.Parameter(torch.Tensor(1, 1, *votes_shape[2:]))
        nn.init.xavier_uniform_(self.dummy_vote)

    def _get_pdf(self, votes, scale):
        pdf = td.Normal(votes, scale)
        return pdf

    def forward(self, x, votes, scales, vote_presence_prob, device, presence=None):
        # x is [B, n_input_points, n_input_dims]    [32, 16, 6]
        batch_size, n_input_points = list(x.shape[:2])

        # votes and scale have shape [B, n_caps, n_input_points, n_input_dims|1]
        # since scale is a per-caps scalar and we have one vote per capsule
        vote_component_pdf = self._get_pdf(votes, torch.unsqueeze(scales, -1))

        # expand along caps dimensions -> [B, 1, n_input_points, n_input_dims]
        expanded_x = torch.unsqueeze(x, 1)
        vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)
        # [B, n_caps, n_input_points]
        vote_log_prob = torch.sum(vote_log_prob_per_dim, -1)
        dummy_vote_log_prob = torch.zeros([batch_size, 1, n_input_points], device=device)
        dummy_vote_log_prob = dummy_vote_log_prob - 2. * math.log(10.)

        # [B, n_caps + 1, n_input_points]
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob], 1)

        # [B, n_caps, n_input_points]
        mixing_logits = safe_log(vote_presence_prob)

        dummy_logit = torch.zeros([batch_size, 1, 1], device=device) - 2. * math.log(10.)
        dummy_logit = dummy_logit.repeat(1, 1, n_input_points)

        # [B, n_caps + 1, n_input_points]
        mixing_logits = torch.cat([mixing_logits, dummy_logit], 1)
        mixing_log_prob = mixing_logits - torch.logsumexp(mixing_logits, 1, keepdim=True)
        # [B, n_input_points]
        mixture_log_prob_per_point = torch.logsumexp(
            mixing_logits + vote_log_prob, 1)

        if presence is not None:
            presence = presence.float()
            mixture_log_prob_per_point = mixture_log_prob_per_point * presence

        # [B,]
        mixture_log_prob_per_example\
            = torch.sum(mixture_log_prob_per_point, 1)

        # []
        mixture_log_prob_per_batch = torch.mean(mixture_log_prob_per_example)

        # [B, n_caps + 1, n_input_points]
        posterior_mixing_logits_per_point = mixing_logits + vote_log_prob

        # [B, n_input_points]
        winning_vote_idx = torch.argmax(
            posterior_mixing_logits_per_point[:, :-1], 1)

        winning_vote = index_select_nd(votes.permute(0, 2, 1, 3), winning_vote_idx)
        winning_pres = index_select_nd(vote_presence_prob.permute(0, 2, 1), winning_vote_idx)
        vote_presence = torch.gt(mixing_logits[:, :-1], mixing_logits[:, -1:])

        # the first four votes belong to the square
        is_from_capsule = winning_vote_idx // self._n_votes

        posterior_mixing_probs = F.softmax(
            posterior_mixing_logits_per_point, 1)

        # [batch_size, 1, self._n_votes, 6]
        dummy_vote = self.dummy_vote.repeat(batch_size, 1, 1, 1)
        dummy_pres = torch.zeros([batch_size, 1, n_input_points], device=device)

        votes = torch.cat((votes, dummy_vote), 1)
        pres = torch.cat([vote_presence_prob, dummy_pres], 1)

        soft_winner = torch.sum(
            torch.unsqueeze(posterior_mixing_probs, -1) * votes, 1)
        soft_winner_pres = torch.sum(
            posterior_mixing_probs * pres, 1)

        posterior_mixing_probs = posterior_mixing_probs[:, :-1].permute(0, 2, 1)

        assert winning_vote.shape == x.shape

        return self.OutputTuple(
            log_prob=mixture_log_prob_per_batch,
            vote_presence=vote_presence.float(),
            winner=winning_vote,
            winner_pres=winning_pres,
            soft_winner=soft_winner,
            soft_winner_pres=soft_winner_pres,
            posterior_mixing_probs=posterior_mixing_probs,
            is_from_capsule=is_from_capsule,
            mixing_logits=mixing_logits,
            mixing_log_prob=mixing_log_prob,
        )


def _capsule_entropy(caps_presence_prob, k=1, **unused_kwargs):
    """Computes entropy in capsule activations."""
    del unused_kwargs

    within_example = normalize(caps_presence_prob, 1)
    within_example = safe_ce(within_example, within_example*k)

    between_example = torch.sum(caps_presence_prob, 0)
    between_example = normalize(between_example, 0)
    between_example = safe_ce(between_example, between_example * k)

    return within_example, between_example


# kl(aggregated_prob||uniform)
def _neg_capsule_kl(caps_presence_prob, **unused_kwargs):
    del unused_kwargs

    num_caps = int(caps_presence_prob.shape[-1])
    return _capsule_entropy(caps_presence_prob, k=num_caps)


# l2(aggregated_prob - constant)
def _caps_pres_l2(caps_presence_prob, num_classes=10.,
                  within_example_constant=0., **unused_kwargs):
    """Computes l2 penalty on capsule activations."""

    del unused_kwargs

    batch_size, num_caps = list(caps_presence_prob.shape)

    if within_example_constant == 0.:
        within_example_constant = float(num_caps) / num_classes

    between_example_constant = float(batch_size) / num_classes

    within_example = l2_loss(
        torch.sum(caps_presence_prob, 1)
        - within_example_constant) / batch_size * 2.

    between_example = l2_loss(
        torch.sum(caps_presence_prob, 0)
        - between_example_constant) / num_caps * 2.

    # neg between example because it's subtracted from the loss later on
    return within_example, -between_example


def sparsity_loss(loss_type, *args, **kwargs):
    """Computes capsule sparsity loss according to the specified type."""

    if loss_type == 'entropy':
        sparsity_func = _capsule_entropy

    elif loss_type == 'kl':
        sparsity_func = _neg_capsule_kl

    elif loss_type == 'l2':
        sparsity_func = _caps_pres_l2

    else:
        raise ValueError(
            'Invalid sparsity loss: "{}"'.format(loss_type))

    return sparsity_func(*args, **kwargs)
