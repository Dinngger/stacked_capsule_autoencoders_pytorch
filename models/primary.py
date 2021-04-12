import numpy as np
import collections
from monty.collections import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import geometric_transform, safe_log


class PrimaryImageEncoder(nn.Module):
    """Primary Image Encoder"""
    OutputTuple = collections.namedtuple(  # pylint:disable=invalid-name
        'PrimaryCapsuleTuple',
        'pose feature presence presence_logit '
        'img_embedding')

    def __init__(self, n_caps, d_pose, d_feature):
        super(PrimaryImageEncoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
        )
        self.embedding_bias = nn.Parameter(torch.Tensor(1, 128, 1, 1))
        nn.init.xavier_uniform_(self.embedding_bias)
        self.n_caps = n_caps
        self.splits = [d_pose, d_feature, 1]
        self.n_dims = sum(self.splits)
        self.conv_enc = nn.Conv2d(128, self.n_dims * n_caps, 1, 1)

    def forward(self, img):
        batch_size = img.size(0)
        img_embedding = self.cnn_encoder(img)
        h = img_embedding + self.embedding_bias
        h = self.conv_enc(h)
        h = h.mean((2, 3))
        h = h.reshape(batch_size, self.n_caps, self.n_dims)
        pose, feature, pres_logit = torch.split(h, self.splits, -1)
        pres = torch.sigmoid(pres_logit)
        pose = geometric_transform(pose)
        return self.OutputTuple(pose, feature, pres, pres_logit, img_embedding)


class TemplateBasedImageDecoder(nn.Module):
    """Template-based primary capsule decoder for images."""

    def __init__(self,
                 output_size,
                 template_size,
                 n_templates):
        super(TemplateBasedImageDecoder, self).__init__()
        self.output_size = output_size
        self.template_size = template_size
        self.n_templates = n_templates
        self.template_logits = nn.Parameter(torch.from_numpy(self.make_templates()))
        self.bg_value = nn.Parameter(torch.Tensor([0]))
        self.temperature_logit = nn.Parameter(torch.Tensor([0]))

    def forward(self, pose, pres, feature, device, img_embedding=None):
        batch_size = pose.size(0)
        batch_shape = list(pose.shape[:2])
        extend_batch_size = np.prod(batch_shape)
        pres = pres.squeeze(-1)

        templates = F.relu(self.template_logits).repeat((batch_size, 1, 1, 1))
        templates = templates.reshape([extend_batch_size, 1] + list(templates.shape[2:]))

        pose = pose.reshape([extend_batch_size] + [2, 3])
        grid = F.affine_grid(pose, [extend_batch_size, 1] + self.output_size, align_corners=True)
        transformed_templates = F.grid_sample(templates, grid, align_corners=True)
        transformed_templates = transformed_templates.reshape(batch_shape + list(transformed_templates.shape[2:]))

        bg_image = torch.zeros_like(transformed_templates[:, :1]) + torch.sigmoid(self.bg_value)
        transformed_templates = torch.cat([transformed_templates, bg_image], 1)

        pres = torch.cat([pres, torch.ones((batch_size, 1), device=device)], 1)
        temperature = F.softplus(self.temperature_logit + .5) + 1e-4
        template_mixing_logits = transformed_templates / temperature

        template_mixing_logits = template_mixing_logits + safe_log(pres).unsqueeze(-1).unsqueeze(-1)
        rec_pdf = (template_mixing_logits, transformed_templates)

        return AttrDict(
            raw_templates=torch.squeeze(F.relu(self.template_logits), 0),
            transformed_templates=transformed_templates[:, :-1],
            mixing_logits=template_mixing_logits[:, :-1],
            pdf=rec_pdf)

    def make_templates(self):
        template_shape = ([1, self.n_templates] + list(self.template_size))
        n_elems = np.prod(template_shape[2:])

        # make each templates orthogonal to each other at init
        n = max(self.n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:self.n_templates, :n_elems].reshape(
            template_shape).astype(np.float32)

        q = (q - q.min()) / (q.max() - q.min())
        return q
