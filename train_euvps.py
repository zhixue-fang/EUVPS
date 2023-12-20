import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
from datetime import datetime
from collections import OrderedDict
from typing import Any, Dict, List, Set
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, build_detection_train_loader

from mask2former import add_maskformer2_config
from vita import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    YTVISEvaluator,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    add_vita_config,
)
from vita.vita_model import Vita


from euvps.euvps_model import EUVPS
from euvps.config import add_euvps_config
from euvps.modeling.sem_seg_head.euvps_sem_seg_head import euvps_sem_seg_head
from euvps.modeling.backbone.res2net50 import build_res2net50
from euvps.modeling.backbone.HRNet import build_hrnet
from euvps.data.SUN_SEG import load_sun_seg_data, build_train_loader, mapper
from euvps.data.CVC_612 import load_cvc_612_training_set, build_train_loader, mapper

class Trainer(DefaultTrainer):
    """"""
    @classmethod
    def build_train_loader(cls, cfg):
        """"""
        return build_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_euvps_config(cfg)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    DatasetCatalog.register("SUN-SEG", load_sun_seg_data)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

