# R-band-chromosome-recognition

## Introduction

This is the code for MICCAI2022 paper "An End-to-End Combinatorial Optimization Method forR-band Chromosome Recognition withGrouping Guided Attention".

## Installation

Dependency: Python3; PyTorch

Download the following pretrained model and put them in model/pretrain folder:

https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

https://download.pytorch.org/models/resnet50-19c8e357.pth

## Usage

For training the method proposed in the paper, please running:
    
    python train.py --model resnet50_gfim_dam --model_name model_resnet50_gfim_dam

For training the baseline ResNet50 model, please running:

    python train.py --model resnet50 --model_name model_resnet50

## Citation

Please cite the following paper if you feel this work is useful to your research

    @inproceedings{2022An,
      title={An End-to-End Combinatorial Optimization Method forR-band Chromosome Recognition withGrouping Guided Attention},    
      author={ Xia, Chao  and  Wang, Jiyue  and  Qin, Yulei  and  Gu, Yun  and  Chen, Bing  and  Yang, Jie },
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      year={2022},
    }

## Contact

For any question, please file an issue or contact

    Chao Xia: xiabc612@gmail.com
