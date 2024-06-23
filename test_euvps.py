import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
try:
    # ignore ShapelyDeprecationWarning from fvcore
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
from PIL import Image
from tqdm import tqdm

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


from euvps.euvps_model import Mynet
from euvps.config import add_mynet_config
from euvps.modeling.sem_seg_head.euvps_sem_seg_head import mynet_sem_seg_head
from euvps.modeling.backbone.res2net50 import build_res2net50
from euvps.modeling.backbone.HRNet import build_hrnet
from euvps.data.SUN_SEG import load_sun_seg_data, build_train_loader, mapper, load_test_sun_seg_data
from euvps.data.CVC_612 import load_cvc_612_training_set, load_cvc_612_test_set, build_train_loader, mapper
from torchvision.transforms import ToTensor as torchtotensor


def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_mynet_config(cfg)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    DatasetCatalog.register("SUN-SEG", load_test_sun_seg_data)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)

    data_dir = cfg.DATA_DIR
    save_dir = cfg.SAVE_DIR
    model_path = cfg.MODEL_PATH

    dataloader = build_train_loader(cfg, mapper)
    model = DefaultTrainer.build_model(cfg)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict["model"])

    model.training = False

    dataset = load_test_sun_seg_data()

    for i in tqdm(range(len(dataset))):
        data = mapper(dataset[i])
        out = model([data])
        result = out.squeeze().cpu().detach().numpy()
        for res, path in zip(result, data["sem_seg_file"]):
            safe_save(Image.fromarray((res * 255).astype(np.uint8)),
                      path.replace(data_dir, save_dir).replace("GT", ""))

        del data
        del out












