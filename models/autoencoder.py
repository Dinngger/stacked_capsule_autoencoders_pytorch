import numpy as np
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.primary import PrimaryImageEncoder, TemplateBasedImageDecoder
from models.attention import SetTransformer
from models.capsule import CapsuleLayer, CapsuleLikelihood, sparsity_loss
from models.utils import flat_reduce, batch_flatten
from models.probe import ClassificationProbe
from models.prob import MixtureDistribution


class ImageCapsule(nn.Module):
    """Object Capsule decoder"""

    def __init__(self, n_caps, d_caps, d_feat, n_votes, **capsule_kwargs):
        super(ImageCapsule, self).__init__()
        self._n_caps = n_caps
        self.d_caps = d_caps
        self._n_votes = n_votes
        self.capsule_layer = CapsuleLayer(n_caps, d_caps, d_feat, n_votes, **capsule_kwargs)
        self.likelihood = CapsuleLikelihood([1, self._n_caps, self._n_votes, 6])

    def forward(self, h, x, device, presence=None):
        """Builds the module.

        Args:
          h: Tensor of encodings of shape [B, n_enc_dims].
          x: Tensor of inputs of shape [B, n_points, n_input_dims]
          presence: Tensor of shape [B, n_points, 1] or None; if it exists, it
            indicates which input points exist.

        Returns:
          A bunch of stuff.
        """
        batch_size = h.size(0)
        res = self.capsule_layer(h, device)
        vote_shape = [batch_size, self._n_caps, self._n_votes, 6]
        res.vote = res.vote[..., :-1, :].reshape(vote_shape)

        votes, scale, vote_presence_prob = res.vote, res.scale, res.vote_presence

        ll_res = self.likelihood(x, votes, scale, vote_presence_prob, device=device, presence=presence)
        res.update(ll_res._asdict())

        caps_presence_prob, _ = torch.max(vote_presence_prob.reshape([batch_size, self._n_caps, self._n_votes]), 2)

        res.caps_presence_prob = caps_presence_prob
        return res


class ImageAutoencoder(nn.Module):
    """Capsule autoencoder"""

    def __init__(
        self,
        config,
        n_classes=10,
        dynamic_l2_weight=10,
        caps_ll_weight=1.,
        img_summaries=False,
        stop_grad_caps_inpt=True,
        stop_grad_caps_target=True,
        prior_sparsity_loss_type='l2',
        prior_within_example_constant=0.,
        posterior_sparsity_loss_type='entropy',
        primary_caps_sparsity_weight=0.,
        weight_decay=0.
    ):
        super(ImageAutoencoder, self).__init__()

        img_size = [config.canvas_size] * 2
        template_size = [config.template_size] * 2

        self._primary_encoder = PrimaryImageEncoder(
            config.n_part_caps,
            config.d_part_pose,
            config.d_part_features)
        self._primary_decoder = TemplateBasedImageDecoder(
            output_size=img_size,
            template_size=template_size,
            n_templates=config.n_part_caps)
        self._encoder = SetTransformer(
            d_x=np.prod(template_size) + config.d_part_features + config.d_part_pose + 1,
            d_h=16,
            d_o=256,
            n_layers=3,
            n_output=config.n_obj_caps,
            n_heads=1)
        self._decoder = ImageCapsule(
            n_caps=config.n_obj_caps,
            d_caps=2,
            d_feat=256,
            n_votes=config.n_part_caps,
            n_caps_params=config.n_obj_caps_params,
            n_hidden=128)
        self._n_classes = n_classes

        self._dynamic_l2_weight = dynamic_l2_weight
        self._caps_ll_weight = caps_ll_weight
        self._img_summaries = img_summaries

        self._stop_grad_caps_inpt = stop_grad_caps_inpt
        self._stop_grad_caps_target = stop_grad_caps_target
        self._prior_sparsity_loss_type = prior_sparsity_loss_type
        self._prior_within_example_sparsity_weight = config.prior_within_example_sparsity_weight
        self._prior_between_example_sparsity_weight = config.prior_between_example_sparsity_weight
        self._prior_within_example_constant = prior_within_example_constant
        self._posterior_sparsity_loss_type = posterior_sparsity_loss_type
        self._posterior_within_example_sparsity_weight = config.posterior_within_example_sparsity_weight
        self._posterior_between_example_sparsity_weight = config.posterior_between_example_sparsity_weight
        self._primary_caps_sparsity_weight = primary_caps_sparsity_weight
        self._weight_decay = weight_decay

        self._classification = ClassificationProbe(n_classes, n_classes)

    def forward(self, img, label, device, labeled=None):
        batch_size = img.size(0)
        primary_caps = self._primary_encoder(img)
        pres = primary_caps.presence

        # expanded_pres = torch.unsqueeze(pres, -1)
        pose = primary_caps.pose
        input_pose = torch.cat([pose, 1. - pres], -1)

        input_pres = pres
        if self._stop_grad_caps_inpt:
            input_pose = input_pose.detach()
            input_pres = input_pres.detach()

        target_pose, target_pres = pose, pres
        if self._stop_grad_caps_target:
            target_pose = target_pose.detach()
            target_pres = target_pres.detach()

        # skip connection from the img to the higher level capsule
        if primary_caps.feature is not None:
            input_pose = torch.cat([input_pose, primary_caps.feature], -1)

        # try to feed presence as a separate input
        # and if that works, concatenate templates to poses
        # this is necessary for set transformer
        # n_templates = primary_caps.pose.size(1)
        templates = F.relu(self._primary_decoder.template_logits)

        inpt_templates = templates
        if self._stop_grad_caps_inpt:
            inpt_templates = inpt_templates.detach()

        if inpt_templates.shape[0] == 1:
            inpt_templates = inpt_templates.repeat((batch_size, 1, 1, 1))
        inpt_templates = batch_flatten(inpt_templates, 2)
        pose_with_templates = torch.cat(
            [input_pose, inpt_templates], -1)

        h = self._encoder(pose_with_templates, presence=input_pres.squeeze(-1))

        res = self._decoder(h, target_pose, device=device, presence=target_pres.squeeze(-1))
        res.primary_presence = primary_caps.presence
        primary_dec_vote = primary_caps.pose
        primary_dec_pres = pres

        # res.bottom_up_rec = self._primary_decoder(
        #     primary_caps.pose,
        #     primary_caps.presence,
        #     feature=primary_caps.feature,
        #     device=device,
        #     img_embedding=primary_caps.img_embedding)

        # res.top_down_rec = self._primary_decoder(
        #     res.winner,
        #     primary_caps.presence,
        #     feature=primary_caps.feature,
        #     device=device,
        #     img_embedding=primary_caps.img_embedding)

        rec = self._primary_decoder(
            primary_dec_vote,
            primary_dec_pres,
            feature=primary_caps.feature,
            device=device,
            img_embedding=primary_caps.img_embedding)

        # tile = res.vote.shape[1]
        # tiled_presence = primary_caps.presence.repeat(tile, 1, 1)

        # tiled_feature = primary_caps.feature
        # if tiled_feature is not None:
        #     tiled_feature = tiled_feature.repeat(tile, 1, 1)

        # tiled_img_embedding = primary_caps.img_embedding.repeat(tile, 1, 1, 1)

        # res.top_down_per_caps_rec = self._primary_decoder(
        #     merge_dims(0, 2, res.vote),
        #     merge_dims(0, 2, res.vote_presence) * tiled_presence.squeeze(-1),
        #     feature=tiled_feature,
        #     device=device,
        #     img_embedding=tiled_img_embedding)

        res.templates = templates
        res.template_pres = pres
        res.used_templates = rec.transformed_templates

        mixing_log_prob = MixtureDistribution.mixing_log_prob(*rec.pdf)
        # res.rec_mode = MixtureDistribution.mode(*rec.pdf, mixing_log_prob)
        # res.rec_mean = MixtureDistribution.mean(*rec.pdf)

        # res.mse_per_pixel = torch.square(img - res.rec_mode)
        # res.mse = flat_reduce(res.mse_per_pixel)

        res.rec_ll_per_pixel = MixtureDistribution.log_prob(*rec.pdf, img, mixing_log_prob)
        res.rec_ll = torch.sum(res.rec_ll_per_pixel) / batch_size

        n_points = int(res.posterior_mixing_probs.shape[1])
        mass_explained_by_capsule = torch.sum(
            res.posterior_mixing_probs, 1)  # (32, 10)

        (res.posterior_within_sparsity_loss,
         res.posterior_between_sparsity_loss) = sparsity_loss(
             self._posterior_sparsity_loss_type,
             mass_explained_by_capsule / n_points,
             num_classes=self._n_classes)

        (res.prior_within_sparsity_loss,
         res.prior_between_sparsity_loss) = sparsity_loss(
             self._prior_sparsity_loss_type,
             res.caps_presence_prob,
             num_classes=self._n_classes,
             within_example_constant=self._prior_within_example_constant)

        if label is not None:
            res.posterior_cls_xe, res.posterior_cls_acc = self._classification(
                mass_explained_by_capsule,
                label,
                labeled)
            res.prior_cls_xe, res.prior_cls_acc = self._classification(
                res.caps_presence_prob,
                label,
                labeled)

        res.best_cls_acc = torch.max(res.prior_cls_acc, res.posterior_cls_acc)
        res.primary_caps_l1 = flat_reduce(res.primary_presence)
        res.weight_decay_loss = 0.0

        return res

    def loss(self, res):

        loss = (-res.rec_ll - self._caps_ll_weight * res.log_prob +
                self._dynamic_l2_weight * res.dynamic_weights_l2 +
                self._primary_caps_sparsity_weight * res.primary_caps_l1 +
                self._posterior_within_example_sparsity_weight *
                res.posterior_within_sparsity_loss -
                self._posterior_between_example_sparsity_weight *
                res.posterior_between_sparsity_loss +
                self._prior_within_example_sparsity_weight *
                res.prior_within_sparsity_loss -
                self._prior_between_example_sparsity_weight *
                res.prior_between_sparsity_loss +
                self._weight_decay * res.weight_decay_loss
                )

        try:
            loss = loss + res.posterior_cls_xe + res.prior_cls_xe
        except AttributeError:
            pass

        return loss
