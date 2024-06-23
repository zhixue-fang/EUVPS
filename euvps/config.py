# -*- coding: utf-8 -*-

from detectron2.config.config import CfgNode as CN

def add_euvps_config(cfg):
    cfg.MODEL.SIZE_DIVISIBILITY = 32
    cfg.MODEL.IN_CHANNEL_DIMS = [(64, 112), (32, 56), (16, 28), (8, 14)]
    cfg.MODEL.SHAPE_DIM = 128
    cfg.MODEL.SHAPE_LAYER = 3
    cfg.MODEL.IN_CHANNELS = [48, 96, 192, 384]
    cfg.MODEL.QUERY_PROC_CHANNELS = [48, 96, 192, 384]
    cfg.MODEL.BACKBONE_FILE = None
    cfg.MODEL.LEN_CLIP_WINDOW = 6
    cfg.MODEL.PER_LEN_CLIP_WINDOW = 3
    cfg.MODEL.SHORT_PROC = False
    cfg.MODEL.BACKBONE.MODEL = None
    cfg.DATA_DIR = "./SUN-SEG/TestEasyDataset/"
    cfg.SAVE_DIR = "./output/vis/"
    cfg.MODEL_PATH = "./output/model_final.pth"


