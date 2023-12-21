from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


def diceCoeffv2(pred, gt, eps=1e-5):
    """"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        bce_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        return bce_loss

class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        mse_loss = F.mse_loss(pred.squeeze(), target.squeeze().float())
        return mse_loss.float()


@META_ARCH_REGISTRY.register()
class EUVPS(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        size_divisibility: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.size_divisibility = size_divisibility

        self.criterion1 = CrossEntropyLoss()
        self.criterion2 = SoftDiceLoss()
        self.criterion3 = MseLoss()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one clip.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (L, C, H, W) format.
                   * "sem_seg": per-image ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different from input resolution), used in inference.
        Returns:
            losses or predictions.
        """
        if self.training:
            return self.train_model(batched_inputs)
        else:
            # NOTE consider only B=1 case.
            return self.inference(batched_inputs)


    def get_targets(self, batched_inputs):
        sem_seg = []
        for video in batched_inputs:
            for frame in video["sem_seg"]:
                sem_seg.append(frame.to(self.device))

        return torch.cat(sem_seg, dim=0)


    def train_model(self, batched_inputs):
        """"""
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        out = self.sem_seg_head([features[key] for key in features.keys()]).permute(1, 0, 2, 3)

        out = F.interpolate(out, size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode="bilinear").sigmoid()

        targets = self.get_targets(batched_inputs).unsqueeze(0).permute(1, 0, 2, 3)

        return {"bce_seg_loss": 5. * self.criterion1(out.contiguous(), targets.contiguous()),
                "dice_seg_loss": 2. * self.criterion2(out.contiguous(), targets.contiguous()),
                "mse_seg_loss": 1. * self.criterion3(out.contiguous(), targets.contiguous())
                }

    def inference(self, batched_inputs):
        """"""
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        out = self.sem_seg_head([features[key] for key in features.keys()]).permute(1, 0, 2, 3)

        out = F.interpolate(out, size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode="bilinear").sigmoid()

        return out


