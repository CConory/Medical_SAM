
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import itertools
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR, WarmupReduceLROnPlateau


def make_optimizer(cfg, model):
    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.gridient_clip_value
        enable = (
                cfg.gradient_clip_enabled
                and cfg.gridient_clip_type == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay

        # different lr schedule
        # if "language_backbone" in key:
        #     lr = cfg.SOLVER.LANG_LR

        # if "backbone.body" in key and "language_backbone.body" not in key:
        #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_BODY_LR_FACTOR

        if "bias" in key:
            lr *= cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias

        if 'norm' in key or 'Norm' in key:
            weight_decay *= cfg.weight_decay_norm_factor
            print("Setting weight decay of {} to {}".format(key, weight_decay))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.optimizer == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.optimizer == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, lr)

    return optimizer

def make_lr_scheduler(cfg, optimizer):

    if cfg.lr_use_cosine:
        max_iters = cfg.max_iter
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters,
            cfg.schedule_gamma,
            warmup_factor=cfg.warmup_factor,
            warmup_iters=cfg.warmup_iters,
            warmup_method=cfg.warmup_method,
            eta_min=cfg.warmup_min_lr
        )

    elif cfg.lr_use_autostep:
        max_iters = cfg.max_iter
        return WarmupReduceLROnPlateau(
            optimizer,
            max_iters,
            cfg.schedule_gamma,
            warmup_factor=cfg.schedule_gamma,
            warmup_iters=cfg.warmup_iters,
            warmup_method=cfg.warmup_method,
            eta_min=cfg.warmup_min_lr,
            patience=cfg.warmup_step_patience,
            verbose=True
        )

    else:
        milestones = []
        for step in cfg.lr_steps:
            if step < 1:
                milestones.append(round(step * cfg.max_iter))
            else:
                milestones.append(step)
        return WarmupMultiStepLR(
            optimizer,
            milestones,
            cfg.schedule_gamma,
            warmup_factor=cfg.schedule_gamma,
            warmup_iters=cfg.warmup_iters,
            warmup_method=cfg.warmup_method,
        )