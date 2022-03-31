# Rethinking Semantic Segmentation: A Prototype View

> Rethinking Semantic Segmentation: A Prototype View,            
> [Tianfei Zhou](https://www.tfzhou.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang/), [Ender Konukoglu](https://scholar.google.com/citations?user=OeEMrhQAAAAJ&hl=en) and [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en) <br>
> *To be appeared in CVPR 2022*

## News
* [2022-03-12] Repo created. Paper and code will come soon.

## Abstract

Prevalent semantic segmentation solutions, despite their different network designs (FCN based or attention based) and mask decoding strategies (parametric softmax based or pixel-query based), can be placed in one category, by con- sidering the softmax weights or query vectors as learnable class prototypes. In light of this prototype view, this study un- covers several limitations of such parametric segmentation regime, and proposes a nonparametric alternative based on non-learnable prototypes. Instead of prior methods learning a single weight/query vector for each class in a fully para- metric manner, our model represents each class as a set of non-learnable prototypes, relying solely on the mean fea- tures of several training pixels within that class. The dense prediction is thus achieved by nonparametric nearest proto- type retrieving. This allows our model to directly shape the pixel embedding space, by optimizing the arrangement be- tween embedded pixels and anchored prototypes. It is able to handle arbitrary number of classes with a constant amount of learnable parameters.We empirically show that, with FCN based and attention based segmentation models (i.e., HR- Net, Swin, SegFormer) and backbones (i.e., ResNet, HRNet, Swin, MiT), our nonparametric framework yields compel- ling results over several datasets (i.e., ADE20K, Cityscapes, COCO-Stuff), and performs well in the large-vocabulary situation. We expect this work will provoke a rethink of the current de facto semantic segmentation model design.

## Citation
```
@inproceedings{zhou2022rethinking,
    author    = {Zhou, Tianfei and Wang, Wenguan and Konukoglu, Ender and Van Gool, Luc},
    title     = {Rethinking Semantic Segmentation: A Prototype View},
    booktitle = {CVPR},
    year      = {2022}
}
```
