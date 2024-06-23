## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional.

### Example conda environment setup
```bash
conda create --name euvps python=3.8 -y
conda activate euvps
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install -U opencv-python

# choose your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/sukjunhwang/VITA.git
cd VITA
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# some operations in mmseg
pip install mmcv-full==1.3.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmcv==1.3.1
cd ~/ # return to your working directory
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .

cd ..
git clone https://github.com/zhixue-fang/EUVPS.git
cd EUVPS
pip install -r requirements.txt
