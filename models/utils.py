import math
import numpy as np
import torch


def geometric_transform(pose_tensor, as_matrix=False):
    scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor, 1, -1)
    scale_x, scale_y = (torch.sigmoid(i) + 1e-2 for i in (scale_x, scale_y))
    trans_x, trans_y, shear = (torch.tanh(i * 5.) for i in (trans_x, trans_y, shear))
    theta = theta * 2. * math.pi
    c, s = torch.cos(theta), torch.sin(theta)
    pose = [
        scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
        trans_x, scale_y * s, scale_y * c, trans_y
    ]
    pose = torch.cat(pose, -1)

    if as_matrix:
        pose = torch.reshape(pose, list(pose.shape[:-1]) + [2, 3])
        zeros = torch.zeros_like(pose[..., :1, 0])
        last = torch.stack([zeros, zeros, zeros + 1], -1)
        pose = torch.cat([pose, last], -2)
    return pose


def safe_log(x, eps=1e-16):
    is_zero = x < eps
    x = torch.where(is_zero, torch.ones_like(x), x)
    x = torch.where(is_zero, torch.zeros_like(x), torch.log(x))
    return x


def safe_ce(labels, probs, axis=-1):
    return torch.mean(-torch.sum(labels * safe_log(probs), dim=axis))


def index_select_nd(input, index, expand=True):
    """
    expand example:
        input: [B, N, C, D]
        index: [B, N]
        out: [B, N, D]
    else example:
        input: [B, N, C, D]
        index: [B, N, X]
        out: [B, N, X, D]
    """
    if expand:
        index = torch.unsqueeze(index, -1)
    ndim_index = len(list(index.shape))
    item_shape = list(input.shape[ndim_index:])
    ndim_item = len(item_shape)
    for _ in range(ndim_item):
        index = torch.unsqueeze(index, -1)
    index = index.repeat([1] * ndim_index + item_shape)
    out = torch.gather(input, ndim_index-1, index)
    if expand:
        out = torch.squeeze(out, ndim_index-1)
    return out


def merge_dims(start, size, x):
    shape = list(x.shape)
    tile = np.prod(shape[start:start+size])
    return x.reshape([tile] + shape[start+size:])


def batch_flatten(x, preserve_dims=1):
    shape = list(x.shape)
    tile = np.prod(shape[preserve_dims:])
    return x.reshape(shape[:preserve_dims] + [tile])


def flat_reduce(x):
    x = batch_flatten(x)
    x = torch.sum(x, -1)
    x = torch.mean(x)
    return x


def normalize(x, axis):
    return x / (torch.sum(x, axis, keepdim=True) + 1e-8)


def l2_loss(x):
    return torch.mean(x ** 2)
