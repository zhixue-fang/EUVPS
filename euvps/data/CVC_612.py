import os
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
import torch
from detectron2.data import DatasetCatalog, build_detection_train_loader
import copy
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import ToTensor as torchtotensor


def load_cvc_612_training_set():
    """"""
    dataset = []

    frame_root = "./cvc612/PNG/Frame/train/"
    gt_root = "./cvc612/PNG/GT/train/"
    size = (256, 448)
    video_time_clips = 9

    img_names = os.listdir(frame_root)
    img_names.sort(key=lambda name: (int(name.split('.')[0])))

    gt_names = os.listdir(gt_root)
    gt_names.sort(key=lambda name: (int(name.split('.')[0])))

    images = [os.path.join(frame_root, x) for x in img_names]
    gts = [os.path.join(gt_root, x) for x in gt_names]

    video_len = len(images)
    clip_len = video_time_clips

    for i in range(video_len // clip_len):
        data_dict = {}
        data_dict["file_name"] = images[i * clip_len:i * clip_len + clip_len]
        data_dict["sem_seg_file_name"] = gts[i * clip_len:i * clip_len + clip_len]
        data_dict["height"] = size[0]
        data_dict["width"] = size[1]
        data_dict["clip_len"] = clip_len
        dataset.append(data_dict)

    dataset.append({"file_name": images[-clip_len:],
                        "sem_seg_file_name": gts[-clip_len:],
                        "height": size[0],
                        "width": size[1],
                        "clip_len": clip_len,
                        })

    return dataset


def load_cvc_612_test_set():
    """"""
    dataset = []

    frame_root = "./cvc612/PNG/Frame/test/"
    gt_root = "./cvc612/PNG/GT/test/"
    size = (256, 448)
    video_time_clips = 9

    img_names = os.listdir(frame_root)
    img_names.sort(key=lambda name: (int(name.split('.')[0])))

    gt_names = os.listdir(gt_root)
    gt_names.sort(key=lambda name: (int(name.split('.')[0])))

    images = [os.path.join(frame_root, x) for x in img_names]
    gts = [os.path.join(gt_root, x) for x in gt_names]

    video_len = len(images)
    clip_len = video_time_clips

    for i in range(video_len // clip_len):
        data_dict = {}
        data_dict["file_name"] = images[i * clip_len:i * clip_len + clip_len]
        data_dict["sem_seg_file_name"] = gts[i * clip_len:i * clip_len + clip_len]
        data_dict["height"] = size[0]
        data_dict["width"] = size[1]
        data_dict["clip_len"] = clip_len
        dataset.append(data_dict)

    dataset.append({"file_name": images[-clip_len:],
                        "sem_seg_file_name": gts[-clip_len:],
                        "height": size[0],
                        "width": size[1],
                        "clip_len": clip_len,
                        })

    return dataset


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    totensor = torchtotensor()
    image = [Image.open(x).convert('RGB').resize((dataset_dict["width"], dataset_dict["height"]), Image.BILINEAR) for x in dataset_dict["file_name"]]
    image = [totensor(x) for x in image]
    label = [Image.open(x).convert('L').resize((dataset_dict["width"], dataset_dict["height"]), Image.NEAREST) for x in dataset_dict["sem_seg_file_name"]]
    label = [totensor(x).long() for x in label]

    return {"image":image,
            "sem_seg":label,
            "width":dataset_dict["width"],
            "height":dataset_dict["height"],
            "image_file":dataset_dict["file_name"],
            "sem_seg_file":dataset_dict["sem_seg_file_name"]}


def build_train_loader(cfg, mapper):
    """"""
    return build_detection_train_loader(cfg, mapper=mapper)


if __name__ == '__main__':
    data = load_cvc_612_training_set()
    print(len(data))