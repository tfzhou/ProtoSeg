# Rethinking Semantic Segmentation: A Prototype View

> Rethinking Semantic Segmentation: A Prototype View,            
> [Tianfei Zhou](https://www.tfzhou.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang/), [Ender Konukoglu](https://scholar.google.com/citations?user=OeEMrhQAAAAJ&hl=en) and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en) <br>
> *CVPR 2022 (Oral) ([arXiv 2203.15102](https://arxiv.org/abs/2203.15102))*

## News
* [2022-04-19] Release the code based on [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch)!
* [2022-03-31] Paper link updated!
* [2022-03-12] Repo created. Paper and code will come soon.

## Abstract

Prevalent semantic segmentation solutions, despite their different network designs (FCN based or attention based) and mask decoding strategies (parametric softmax based or pixel-query based), can be placed in one category, by considering the softmax weights or query vectors as learnable class prototypes. In light of this prototype view, this study uncovers several limitations of such parametric segmentation regime, and proposes a nonparametric alternative based on non-learnable prototypes. Instead of prior methods learning a single weight/query vector for each class in a fully parametric manner, our model represents each class as a set of non-learnable prototypes, relying solely on the mean features of several training pixels within that class. The dense prediction is thus achieved by nonparametric nearest prototype retrieving. This allows our model to directly shape the pixel embedding space, by optimizing the arrangement between embedded pixels and anchored prototypes. It is able to handle arbitrary number of classes with a constant amount of learnable parameters.We empirically show that, with FCN based and attention based segmentation models (i.e., HR-Net, Swin, SegFormer) and backbones (i.e., ResNet, HRNet, Swin, MiT), our nonparametric framework yields compelling results over several datasets (i.e., ADE20K, Cityscapes, COCO-Stuff), and performs well in the large-vocabulary situation. We expect this work will provoke a rethink of the current de facto semantic segmentation model design.

## Installation

This implementation is built on [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch). Many thanks to the authors for the efforts.

Please follow the [Getting Started](https://github.com/openseg-group/openseg.pytorch/blob/master/GETTING_STARTED.md) for installation and dataset preparation.

## Performance

### Cityscapes

| Method | Train Set | Val Set | Iters | Batch Size | mIoU  | Log | CKPT | Script |
| --------- | ---------- | --------- | ------- | ---------- | ----- | --- | ----   | ----  |
| HRNet  | train     | val     | 80K   | 8          | 79.0  | [log](https://github.com/tfzhou/pretrained_weights/releases/download/v.cvpr22/213158796.out) | [ckpt](https://github.com/tfzhou/pretrained_weights/releases/download/v.cvpr22/hrnet_w48_lr1x_hrnet_ce_80k_latest.pth) |```scripts/cityscapes/hrnet/run_h_48_d_4.sh```|
| Ours   | train     | val     | 80K   | 8          | 80.1  | [log](https://github.com/tfzhou/pretrained_weights/releases/download/v.cvpr22/214330916.out) | [ckpt](https://github.com/tfzhou/pretrained_weights/releases/download/v.cvpr22/hrnet_w48_proto_lr1x_hrnet_proto_80k_latest.pth) |```scripts/cityscapes/hrnet/run_h_48_d_4_proto.sh```|

_More results will come soon_

## Citation
```
@inproceedings{zhou2022rethinking,
    author    = {Zhou, Tianfei and Wang, Wenguan and Konukoglu, Ender and Van Gool, Luc},
    title     = {Rethinking Semantic Segmentation: A Prototype View},
    booktitle = {CVPR},
    year      = {2022}
}
```

## Relevant Projects

> Please also see our works [1] for a novel training paradigm with a **cross-image, pixel-to-pixel contrative loss**, 
> and [2] for a novel **hierarchy-aware segmentation learning scheme** for structured scene parsing.

[1] Exploring Cross-Image Pixel Contrast for Semantic Segmentation - ICCV 2021 (Oral) [[arXiv](https://arxiv.org/abs/2101.11939)][[code](https://github.com/tfzhou/ContrastiveSeg)]

[2] Deep Hierarchical Semantic Segmentation - CVPR 2022 [[arXiv](https://arxiv.org/abs/2203.14335)][[code](https://github.com/0liliulei/HieraSeg)]

