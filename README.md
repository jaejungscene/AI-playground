# ML & DL Paper List

- [Computer Vision](#computer-vision)
  - [Image Classification](#image-classification)
  - [Action Classificatin](#action-classification)
  - [Object Detection](#object-detection)
  - [Image Segmentation](#image-segmentation)
  - [Image Super Resolution](#image-super-resolution)
  - [Domain Generalization](#domain-generalization)
- [Util](#util)
- [Semi Supervised Learning](#semi-supervised-learning)
  - [Computer Vision](#computer-vision)
- [NLP](#nlp)

<br>

-------------------------------------------------------

<br>

# Computer Vision

### Image Classification
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[AlexNet](https://arxiv.org/abs/1512.03385) | NeurIPS 2012 | [korean](https://mountain96.tistory.com/33) | [notion](https://voltaic-chipmunk-57d.notion.site/AlexNet-Deep-Residual-Learning-for-Image-Recognition-0326ff266aa04d59b0202f1f4bfe4cae) | -
[Vgg](https://arxiv.org/pdf/1409.1556.pdf) | ICLR 2015 | - | [notion](https://voltaic-chipmunk-57d.notion.site/Vgg-Very-Deep-Convolutional-Networks-for-Large-scale-Image-Recognition-b1e089011bc7480f916ced0cb186cf9f) | -
[GoogleNet](https://arxiv.org/pdf/1409.4842v1.pdf) | CVPR 2015 | [korean](https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0) | - | [Inception v1 block](https://github.com/jaejungscene/AI_read-paper-list/blob/main/code/inceptionV1.py)
[ResNet](https://arxiv.org/pdf/1512.03385.pdf) | CVPR 2015 | - | - | -
[Inception v2,v3](https://paperswithcode.com/paper/rethinking-the-inception-architecture-for) | CVPR 2016 | [korean](https://gaussian37.github.io/dl-concept-inception/#:~:text=%EC%95%9E%EC%97%90%EC%84%9C%20%EC%84%A4%EB%AA%85%ED%95%9C%20%EA%B2%83%EA%B3%BC%20%EA%B0%99%EC%9D%B4%201xn,%ED%9B%84%EA%B0%80%20%EB%8C%80%EC%9D%91%EC%9D%B4%20%EB%90%A9%EB%8B%88%EB%8B%A4) | -
[ResNext](https://arxiv.org/pdf/1611.05431.pdf) | CVPR 2017 | - | - | -
[WideResNet](https://paperswithcode.com/method/wideresnet) | 2017 | - | - 
[MobileNet](https://arxiv.org/pdf/1704.04861.pdf) | 2017 | - | - | [Depthwise Separable Conv block](https://github.com/jaejungscene/AI_read-paper-list/blob/main/code/DepthwiseSeparableConv.py)
[MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) | CVPR 2018 | - | - | [Inverted Residual block](https://github.com/jaejungscene/AI_read-paper-list/blob/9d6ad804fb9b0f2db548719792bacfef77d1057e/code/simple_mobilenetV3.py#L22)
[SENet](https://arxiv.org/pdf/1709.01507.pdf) | CVPR 2018 | - | - | [SE block](https://github.com/jaejungscene/read-paper-list/blob/main/code/seblock.py)
[CBAM](https://arxiv.org/pdf/1807.06521.pdf) | ECCV 2018 | - | - | [CBAM block](https://github.com/jaejungscene/read-paper-list/blob/main/code/cbam.py)
[MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf) | CVPR 2019 | - | - | [hswish, simple mobilenetV3](https://github.com/jaejungscene/AI_read-paper-list/blob/e38afeba877691aaeeb808944435a9797a73eff5/code/simple_mobilenetV3.py#L1)
[EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) | ICML 2019 | - | - | -
[RegNet](https://arxiv.org/abs/2003.13678) | CVPR 2020 | [korean](https://2-chae.github.io/category/2.papers/31) | - | -
[ViT](https://arxiv.org/pdf/2010.11929.pdf) | ICLR 2021 | [korean](https://gaussian37.github.io/dl-concept-vit/) | - | -
[DeiT](https://arxiv.org/abs/2012.12877) | ICML 2021 | - | - | -

### Action Classification
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[Non-local Neural Net](https://paperswithcode.com/paper/non-local-neural-networks) | CVPR 2018 | [korean](https://blog.lunit.io/2018/01/19/non-local-neural-networks/) | - | - | -

### Object Detection
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[RCNN](https://arxiv.org/abs/1311.2524) | 2014 | - | - | -
[Fast-RCNN](http://arxiv.org/abs/1504.08083) | 2015 | - | - | -
[Faster-RCNN](http://arxiv.org/abs/1506.01497) | NeurIPS 2015 | - | - | -
[Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf) | 2017 | - | - | -
[FPN](https://arxiv.org/abs/1612.03144) | 2017 | - | - | -
[YOLO](http://arxiv.org/abs/1506.02640) | 2016 | - | - | -
[SSD](http://arxiv.org/abs/1512.02325) | 2016 | - | - | -

### Image Segmentation
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[U-Net](https://arxiv.org/pdf/1505.04597.pdf) | 2015 | - | - | -

### Image Super Resolution
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[Pixel Shuffle](https://paperswithcode.com/method/pixelshuffle) | 2016 | - | - | -
[SRCNN](https://arxiv.org/abs/1501.00092)| - | - | - | -
[SRGAN](https://arxiv.org/abs/1609.04802)| - | - | - | -

### Domain Generalization
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification](https://arxiv.org/pdf/2012.00417.pdf) | CVPR 2021 | - | [highlighted](https://github.com/jaejungscene/AI_read-paper-list) | -

<br>
<br>

# Util
NAME | PUBLISHED IN | FIELD | MY REVIEW | CODE
-- | -- | -- | -- | --
[Mixup](https://arxiv.org/pdf/1710.09412.pdf) | 2018 | Data Augmentation | - | [code](https://github.com/jaejungscene/AI_read-paper-list/blob/0db9c4b346ae68a34bb2c15d4e0cddd3a9427c7d/code/mixup.py#L1)
[Cutmix](https://arxiv.org/pdf/1905.04899.pdf) | 2019 | Data Augmentation | - | [code](https://github.com/jaejungscene/AI_read-paper-list/blob/0db9c4b346ae68a34bb2c15d4e0cddd3a9427c7d/code/cutmix.py#L1)
[AutoAugment](https://arxiv.org/pdf/1805.09501v3.pdf) | 2019 | Data Augmentation | - | -
[RandAugment](https://arxiv.org/pdf/1909.13719.pdf) | 2019 | Data Augmentation | - | -
[Temperature Scaling](https://paperswithcode.com/paper/on-calibration-of-modern-neural-networks) | ICML 2017 | Learning Method | - | -
[Label Smoothing](https://arxiv.org/abs/1906.02629) | 2019 | Learning Method | - | -
[ArcFace](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep), [Updated ArcFace](https://arxiv.org/abs/1801.07698) | CVPR2019, 2022 | Loss Function & Face Recognition | [highlighted](https://github.com/jaejungscene/AI-PAPER-LIST/blob/main/mypdf/ArcFace.pdf) | [code](https://github.com/jaejungscene/AI_read-paper-list/blob/e97b18363333ee127b39a2e183bf6f4e891555ce/code/ArcfaceLoss.py#L4)

<br>
<br>

# Semi Supervised Learning

### Computer Vision
NAME | PUBLISHED IN | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | --
[Pseudo Label](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks) | 2013 | [korean](https://deep-learning-study.tistory.com/553) | - | -
[Noisy Student](https://arxiv.org/abs/1911.04252) | CVPR 2020 | [korean](https://2-chae.github.io/category/2.papers/24) | - | -
[Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) | CVPR 2021 | [korean](https://kmhana.tistory.com/33) | - | -

<br>
<br>

# NLP
NAME | PUBLISHED IN | FIELD | OTHER'S REVIEW | MY REVIEW | CODE
-- | -- | -- | -- | -- | --
[CBoW & skip-gram](https://arxiv.org/abs/1301.3781v3) | - | Word2Vec | - | - | [only python](https://github.com/jaejungscene/Deep_Learning_from_Scratch/tree/main/Volume_2/ch04)
