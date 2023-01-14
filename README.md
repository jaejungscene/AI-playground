# Awesome AI ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
This page will continue to be updated.  
A curated list of resources is dedicated to every AI topics.  
Feel free to pull requests.  


## Table of Contents
- [Image Classification](#image-classification)
- [Action Classificatin](#action-classification)
- [Object Detection](#object-detection)
- [Image Segmentation](#image-segmentation)
- [Image Super Resolution](#image-super-resolution)
- [Face Recognition](#face-recognition)
- [Meta Learning](#meta-learning)
- [Domain Generalization](#domain-generalization)
- [Semi Supervised Learning](#semi-supervised-learning)
- [Data Augmentation](#data-augmentation)
- [Natural Language Processing](#natural-language-processing)
- [Util](#util)

<br>

-------------------------------------------------------

<br>

## Image Classification
Year | Name | Paper | Code
-- | -- | -- | -- |
|1998|LeNet: Gradient-based learning applied to document recognition|[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb)|
|2012|AlexNet: ImageNet Classification with Deep Convolutional Neural Networks|[PDF](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_convolutional-modern/alexnet.ipynb) |
|2014|VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition|[PDF](https://arxiv.org/pdf/1409.1556.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/datastrophy/vgg16-pytorch-implementation) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py)|
|2015|GoogLeNet: Going Deeper with Convolutions|[PDF](https://arxiv.org/pdf/1409.4842.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ashfakyeafi/googlenet-from-scratch-pytorch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Lornatang/GoogLeNet-PyTorch/blob/main/model.py)|
|2016|ResNet: Deep Residual Learning for Image Recognition|[PDF](https://arxiv.org/abs/1512.03385)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/won6314/resnet-with-pytorch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)|
|2017|SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size|[PDF](https://arxiv.org/abs/1602.07360)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/kerimelebiler/digit-recognizer-with-squeezenet) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/forresti/SqueezeNet)|
|2017|DenseNet: Densely Connected Convolutional Networks|[PDF](https://arxiv.org/abs/1608.06993)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/kevfern/densenet-implementation-for-classification#3.-DenseNet) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/liuzhuang13/DenseNet)|
|2017|XceptionNet: Deep Learning with Depthwise Separable Convolutions|[PDF](https://arxiv.org/abs/1610.02357)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/yasserh/xception-implementation) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tstandley/Xception-PyTorch)|
|2018|MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Application|[PDF](https://arxiv.org/abs/1704.04861)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sonukiller99/mobilenet-from-scratch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py)|
|2018|ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices|[PDF](https://arxiv.org/abs/1707.01083)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/dcosmin/shufflenet-with-keras) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jaxony/ShuffleNet)|
|2018|MobileNetV2: Inverted Residuals and Linear Bottlenecks|[PDF](https://arxiv.org/pdf/1801.04381.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/vitouphy/mobilenet-from-scratch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)|
|2018|NASNet: Learning Transferable Architectures for Scalable Image Recognition|[PDF](https://arxiv.org/pdf/1707.07012.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/l4morak/just-a-nasnet) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/MarSaKi/nasnet)|
|2018|Squeeze Excitation Network: Squeeze-and-Excitation Networks|[PDF](https://arxiv.org/pdf/1709.01507.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/longyunguo/cnn-with-a-squeeze-and-excitation-mechanism) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/hujie-frank/SENet)|
|2018|Residual Attention Network: Residual Attention Network for Image Classification|[PDF](https://arxiv.org/pdf/1704.06904.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/chanhu/residual-attention-network-pytorch/notebook) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)|
|2019|CBAM: Convolutional Block Attention Module|[PDF](https://arxiv.org/pdf/1807.06521.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/marcberghouse/cnn-with-cbam-module) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/elbuco1/CBAM)|
|2019|MobileNetV3: Searching for MobileNetV3|[PDF](https://arxiv.org/pdf/1905.02244v5.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/xiaolai-sqlai/mobilenetv3/blob/adc0ca87e1dd8136cd000ae81869934060171689/mobilenetv3.py#L75)|
|2020|RegeNet: Designing Network Design Spaces|[PDF](https://arxiv.org/pdf/2003.13678.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/signatrix/regnet/blob/master/src/regnet.py#L9)|
|2021|EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks|[PDF](https://arxiv.org/pdf/1905.11946.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/vikramsandu/efficientnet-from-scratch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/lukemelas/EfficientNet-PyTorch)|
|2021|Vision Transformer: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale|[PDF](https://arxiv.org/pdf/2010.11929.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/manabendrarout/vision-transformer-vit-pytorch-on-tpus-train/notebook) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/vision_transformer)|
|2021|DeiT: Training data-efficient image transformers & distillation through attention|[PDF](https://arxiv.org/pdf/2012.12877.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/deit/blob/main/models.py#L20)|
|2021|Swin Transformer: Hierarchical Vision Transformer using Shifted Windows|[PDF](https://arxiv.org/pdf/2103.14030.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/pdochannel/swin-transformer-in-pytorch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/Swin-Transformer)|
|2022|ConvNeXt: A ConvNet for the 2020s|[PDF](https://arxiv.org/pdf/2201.03545.pdf)| [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/kalelpark/review-a-convnet-for-the-2020s) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/ConvNeXt)|
|2023|ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders|[PDF](https://arxiv.org/pdf/2301.00808.pdf)| [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/ConvNeXt-V2)|


<br>
<br>

## Action Classification
Year | Name | Paper | Code
-- | -- | -- | -- |
|2018|Non-local Neural Networks|[PDF](https://paperswithcode.com/paper/non-local-neural-networks)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tea1528/Non-Local-NN-Pytorch)

<br>
<br>

## Object Detection
Year | Name | Paper | Code
-- | -- | -- | -- |
|2013|R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation|[PDF](https://arxiv.org/pdf/1311.2524.pdf)| [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/object-detection-algorithm/R-CNN)|
|2015|Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks|[PDF](https://arxiv.org/pdf/1506.01497.pdf)| [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/lonewolf45/faster-rcnn-for-multiclass-object-detection) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jwyang/faster-rcnn.pytorch)|
|2016|OHEM: Training Region-based Object Detectors with Online Hard Example Mining|[PDF](https://arxiv.org/abs/1604.03540)| [![GitHub](https://badges.aleen42.com/src/github.svg)](https://gist.github.com/erogol/c37628286f8efdb62b3cc87aad382f9e)|
|2016|YOLOv1: You Only Look Once: Unified, Real-Time Object Detection|[PDF](https://arxiv.org/pdf/1506.02640.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Tshzzz/pytorch_yolov1)|
|2016|SSD(Single Shot Detection): Single Shot MultiBox Detector|[PDF](https://arxiv.org/pdf/1506.01497.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aman10kr/face-mask-detection-using-ssd) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/amdegroot/ssd.pytorch)|
|2017|FPN(Feature Pyramids Network): Feature Pyramid Networks for Object Detection|[PDF](https://arxiv.org/pdf/1506.01497.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jwyang/fpn.pytorch)|
|2017|RetinaNet: Focal loss for dense object detection|[PDF](https://arxiv.org/pdf/1708.02002v2.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/jainamshah17/gwd-retinanet-pytorch-train) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/yhenon/pytorch-retinanet)|
|2017|Mask-RCNN: Mask R-CNN|[PDF](http://cn.arxiv.org/pdf/1703.06870.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/robinteuwens/mask-rcnn-detailed-starter-code) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/Detectron)|
|2018|YOLOv3: An Incremental Improvement|[PDF](http://cn.arxiv.org/pdf/1804.02767.pdf)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ultralytics/yolov3) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/ayooshkathuria/pytorch-yolo-v3)|
|2018|RefineDet: Single-Shot Refinement Neural Network for Object Detection|[PDF](http://cn.arxiv.org/pdf/1711.06897.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/sfzhang15/RefineDet)|
|2018|M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network|[PDF](https://qijiezhao.github.io/imgs/m2det.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/VDIGPKU/M2Det)|
|2019|Mask scoring r-cnn|[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/zjhuang22/maskscoring_rcnn)|
|2019|FSFA: Feature selective anchor-free module for single-shot object detection|[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/hdjang/Feature-Selective-Anchor-Free-Module-for-Single-Shot-Object-Detection)|
|2019|Scratchdet: Exploring to train single-shot object detectors from scratch|[PDF](https://arxiv.org/pdf/1810.08425.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/KimSoybean/ScratchDet)|

<br>
<br>

## Image Segmentation
#### Segmantic Segmentation
###### Medical(2D, 3D image)
Year | Name | Paper | Code
-- | -- | -- | -- |
|2015|U-Net: Convolutional Networks for Biomedical Image Segmentation|[PDF](https://arxiv.org/pdf/1505.04597.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/milesial/Pytorch-UNet)|
|2021|TransUnet: Transformers make Strong Encoders for Medical Image Segmentation|[PDF](https://arxiv.org/pdf/2102.04306.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Beckschen/TransUNet)|
|2021|Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation|[PDF](https://arxiv.org/pdf/2105.05537.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/HuCaoFighting/Swin-Unet)|
|2021|UNETR: Transformers for 3D Medical Image Segmentation|[PDF](https://arxiv.org/pdf/2103.10504.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)

<br>
<br>

## Image Super Resolution
Year | Name | Paper | Code
-- | -- | -- | -- |
|2014|SRCNN: Image Super-Resolution Using Deep Convolutional Networks|[PDF](https://arxiv.org/abs/1501.00092)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/yjn870/SRCNN-pytorch)|
|2016|Pixel Shuffle: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network|[PDF](https://arxiv.org/pdf/1609.05158v2.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/pytorch/pytorch/blob/460970483d51006c61d504fb27985c60a3efbcd3/torch/nn/modules/pixelshuffle.py#L7)|
|2017|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial|[PDF](https://arxiv.org/pdf/1609.04802.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/leftthomas/SRGAN)|

<br>
<br>

## Face Recognition
Year | Name | Paper | Code
-- | -- | -- | -- |
|2015|FaceNet: A Unified Embedding for Face Recognition and Clustering|[PDF](https://arxiv.org/pdf/1503.03832.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tbmoon/facenet)|
|2018|SphereFace: Deep Hypersphere Embedding for Face Recognition|[PDF](https://arxiv.org/pdf/1704.08063.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/clcarwin/sphereface_pytorch)|
|2018|CosFace: Large Margin Cosine Loss for Deep Face Recognition|[PDF](https://arxiv.org/pdf/1801.09414.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/MuggleWang/CosFace_pytorch)|
|2018|ArcFace: Additive Angular Margin Loss for Deep Face Recognition|[PDF](https://arxiv.org/pdf/1801.07698.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/ronghuaiyang/arcface-pytorch)|

<br>
<br>

## Meta Learning
Year | Name | Paper | Code
-- | -- | -- | -- |
|2015|SiameseNet: Siamese Neural Networks for One-shot Image Recognition|[PDF](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/SoongE/siamese-one-shot-pytorch)|
|2016|MatchingNet: Matching Networks for One Shot Learning|[PDF](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/main/matching_network.py)|

<br>
<br>

## Domain Generalization
#### Computer Vision
Year | Name | Paper | Code
-- | -- | -- | -- |
|2021|Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification|[PDF](https://arxiv.org/pdf/2012.00417.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/HeliosZhao/M3L)|

<br>
<br>

## Semi Supervised Learning
#### Computer Vision
Year | Name | Paper | Code
-- | -- | -- | -- |
|2020|Noisy Student: Self-training with Noisy Student improves ImageNet classification|[PDF](https://arxiv.org/abs/1911.04252.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/HeliosZhao/M3L)|
|2021|Meta Pseudo Labels|[PDF](https://arxiv.org/pdf/2003.10580.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/kekmodel/MPL-pytorch)|

<br>
<br>

## Data Augmentation
#### Computer Vision
Year | Name | Paper | Code
-- | -- | -- | -- |
|2018|Mixup: Beyond Empirical Risk Minimization|[PDF](https://arxiv.org/pdf/1710.09412.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jaejungscene/AI_read-paper-list/blob/0db9c4b346ae68a34bb2c15d4e0cddd3a9427c7d/code/mixup.py#L1)
|2019|Cutmix: Regularization Strategy to Train Strong Classifiers with Localizable Features|[PDF](https://arxiv.org/pdf/1905.04899.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jaejungscene/AI_read-paper-list/blob/0db9c4b346ae68a34bb2c15d4e0cddd3a9427c7d/code/cutmix.py#L1)
|2019|AutoAugment: Learning Augmentation Strategies from Data|[PDF](https://arxiv.org/pdf/1805.09501v3.pdf)|-|
|2019|RandAugment: Practical automated data augmentation with a reduced search space|[PDF](https://arxiv.org/pdf/1909.13719.pdf)|-|

<br>
<br>

## Natural Language Processing
Year | Name | Paper | Code
-- | -- | -- | -- |
|2013|CBoW & skip-gram: Efficient Estimation of Word Representations in Vector Space|[PDF](https://arxiv.org/pdf/1301.3781v3.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jaejungscene/Deep_Learning_from_Scratch/tree/main/Volume_2/ch04)
|2017|Transformer: Attention Is All You Need|[PDF](https://arxiv.org/pdf/1706.03762.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L128)

<br>
<br>

## Util
Year | Name | Paper | Code
-- | -- | -- | -- |
|2017|Temperature Scaling: On Calibration of Modern Neural Networks|[PDF](https://arxiv.org/pdf/1706.04599v2.pdf)|[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/gpleiss/temperature_scaling)
|2020|Label Smoothing: When Does Label Smoothing Help|[PDF](https://arxiv.org/pdf/1906.02629.pdf)|-|
