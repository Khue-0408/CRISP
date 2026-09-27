# PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI2020-Oral & MICCAI2025 Young Scientist Publication Impact Award)

> **Authors:** 
> [Deng-Ping Fan](https://dengpingfan.github.io/), 
> [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
> [Tao Zhou](https://taozh2017.github.io/),
> [Geng Chen](https://www.researchgate.net/profile/Geng_Chen13), 
> [Huazhu Fu](http://hzfu.github.io/), 
> [Jianbing Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing), and 
> [Ling Shao](http://www.inceptioniai.org/).


- Honored to be selected as a [MICCAI 2025 YSPIA Awardee](https://miccai.org/index.php/about-miccai/awards/young-scientist-impact-award/) ![image](https://github.com/user-attachments/assets/3966a8b1-85e9-4398-88a0-108816f78a91) ![image](https://github.com/user-attachments/assets/9dfa8007-af53-4213-9bec-e548fe159261)


- 💥 We’re excited to introduce **PraNet-V2**, bringing semantic segmentation capabilities! PraNet-V2 introduces the **Dual-Supervised Reverse Attention (DSRA)** module, enabling explicit background supervision, independent background modeling, and semantically enriched attention fusion. We have found that PraNet-V2 outperforms PraNet-V1 in **polyp segmentation** and **improves existing SOTA segmentation multi-class models through DSRA integration**. 📖 [Read the paper](https://arxiv.org/abs/2504.10986) | 🔗 [Check out the code](https://github.com/ai4colonoscopy/PraNet-V2)

- :boom: We’re excited to introduce “[IntelliScope Project](https://github.com/ai4colonoscopy/IntelliScope),” which offers a deep dive into the latest advancements in intelligent colonoscopy (📖 **ColonSurvey**). We’re also pushing for three key initiatives to embrace the multimodal era in colonoscopy: a pioneering large-scale instruction tuning dataset (🏥 **ColonINST**), a colonoscopy-specific multimodal language model (🤖 **ColonGPT**), and a **multimdoal benchmark** 💯 for comparing different approaches.

- We are in the MICCAI2024 Young Scientist Publication Impact Award Shortlist
    ![image](https://github.com/user-attachments/assets/f2871d49-235c-4554-ae89-11740e41edb8)
  
- We receive the award of Jittor Developer Conference Distinguish Paper & Most Influential (Application) Paper
    <p align="center">
        <img src="imgs/PraNet-Award.png"/> <br />
    </p>


## 1. Preface

- This repository provides code for "_**PraNet: Parallel Reverse Attention Network for Polyp Segmentation**_" MICCAI-2020. 
([paper](https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_26) | [中文版](https://dengpingfan.github.io/papers/[2020][MICCAI]PraNet_Chinese.pdf))

- If you have any questions about our paper, feel free to contact me. And if you are using PraNet 
or evaluation toolbox for your research, please cite this paper ([BibTeX](#4-citation)).


### 1.1. :fire: NEWS :fire:

- [2025/03/20] 🚀 **PraNet-V2 is here!** 🚀  
  We have introduced **PraNet-V2**, an enhanced version of PRaNet with **Dual-Supervised Reverse Attention (DSRA)** for more effective **multi-class** segmentation.  Check out the paper and code here: [PraNet-V2](https://github.com/ai4colonoscopy/PraNet-V2)

- [2022/11/26] Our PraNet has been developed on [Huawei Ascend platform](https://e.huawei.com/hk/products/servers/ascend), where the project could be found at [Gitee](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/semantic_segmentation/PraNet) and [CSDN introduction](https://blog.csdn.net/m0_62401440/article/details/125563697).

- [2022/03/27] :boom: We release a new large-scale dataset on **Video Polyp Segmentation (VPS)** task, please enjoy it. [ProjectLink](https://github.com/GewelsJI/VPS)/ [PDF](https://arxiv.org/abs/2203.14291).

- [2021/12/26] :boom: PraNet模型在[Jittor Developer Conference 2021](https://cg.cs.tsinghua.edu.cn/jittor/news/2021-12-27-15-27-00-00-jdc1/)中荣获「最具影响力计图论文（应用）奖」

- [2021/09/07] The Jittor convertion of PraNet ([inference code](https://github.com/DengPingFan/PraNet/tree/master/jittor)) is available right now. It has robust inference efficiency compared to PyTorch version, please enjoy it. Many thanks to Yu-Cheng Chou for the excellent conversion from pytorch framework.

- [2021/09/05] The Tensorflow (Keras) implementation of PraNet (ResNet50/MobileNetV2 version) is released in [github-link](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation). Thanks Tauhid Khan.

- [2021/08/18] Improved version (PraNet-V2) has been released: https://github.com/DengPingFan/Polyp-PVT.

- [2021/04/23] We update the results on four [Camouflaged Object Detection (COD)](https://github.com/DengPingFan/SINet) testing datasets (i.e., COD10K, NC4K, CAMO, and CHAMELEON) of our PraNet. Consult the [upstream PraNet repository](https://github.com/DengPingFan/PraNet) for current result and model availability.

- [2021/01/21] :boom: Our PraNet has been used as the base segmentation model of [Prof. Michael I. Jordan](https://scholar.google.com/citations?user=yxUduqMAAAAJ&hl=zh-CN) et al's recent work (Distribution-Free, Risk-Controlling Prediction Sets, [Journal of the ACM 2021](https://arxiv.org/pdf/2101.02703.pdf)).

- [2021/01/10] :boom: Our PraNet achieved the Top-1 ranking on the camouflaged object detection task ([link](https://paperswithcode.com/paper/pranet-parallel-reverse-attention-network-for)). 

- [2020/09/18] Upload the pre-computed maps.

- [2020/05/28] Upload pre-trained weights.

- [2020/06/24] Release training/testing code.

- [2020/03/24] Create repository.


### 1.2. Table of Contents

- [PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI2020-Oral)](#pranet-parallel-reverse-attention-network-for-polyp-segmentation-miccai2020-oral)
  - [1. Preface](#1-preface)
    - [1.1. :fire: NEWS :fire:](#11-fire-news-fire)
    - [1.2. Table of Contents](#12-table-of-contents)
    - [1.3. State-of-the-art Approaches](#13-state-of-the-art-approaches)
  - [2. Overview](#2-overview)
    - [2.1. Introduction](#21-introduction)
    - [2.2. Framework Overview](#22-framework-overview)
    - [2.3. Qualitative Results](#23-qualitative-results)
  - [3. Proposed Baseline](#3-proposed-baseline)
    - [3.1. Training/Testing](#31-trainingtesting)
    - [3.2 Evaluating your trained model:](#32-evaluating-your-trained-model)
    - [3.3 Pre-computed maps:](#33-pre-computed-maps)
  - [4. Citation](#4-citation)
  - [5. TODO LIST](#5-todo-list)
  - [6. FAQ](#6-faq)
  - [7. License](#7-license)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### 1.3. State-of-the-art Approaches  
1. "Selective feature aggregation network with area-boundary constraints for polyp segmentation." IEEE Transactions on Medical Imaging, 2019.
paper link: https://link.springer.com/chapter/10.1007/978-3-030-32239-7_34 
2. "PraNet: Parallel Reverse Attention Network for Polyp Segmentation" IEEE Transactions on Medical Imaging, 2020.
paper link: https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_26
3. "Hardnet-mseg: A simple encoder-decoder polyp segmentation neural network that achieves over 0.9 mean dice and 86 fps" arXiv, 2021
paper link: https://arxiv.org/pdf/2101.07172.pdf
4. "TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation" arXiv, 2021.
paper link: https://arxiv.org/pdf/2102.08005.pdf
5. "Automatic Polyp Segmentation via Multi-scale Subtraction Network" MICCAI, 2021. paper link: https://arxiv.org/pdf/2108.05082.pdf
6. "CCBANet: Cascading Context and Balancing Attention for Polyp Segmentation" MICCAI, 2021. paper link: https://link.springer.com/book/10.1007/978-3-030-87193-2?noAccess=true
7. "Double Encoder-Decoder Networks for Gastrointestinal Polyp Segmentation" MICCAI, 2021. paper link: https://arxiv.org/pdf/2110.01939.pdf
8. "HRENet: A Hard Region Enhancement Network for Polyp Segmentation" MICCAI, 2021. paper link: https://link.springer.com/book/10.1007/978-3-030-87193-2?noAccess=true
9. "Learnable Oriented-Derivative Network for Polyp Segmentation" MICCAI, 2021. paper link: https://link.springer.com/book/10.1007/978-3-030-87193-2?noAccess=true
10. "Shallow attention network for polyp segmentation" MICCAI, 2021. paper link: https://arxiv.org/pdf/2108.00882.pdf

The latest trends in image-/video-based polyp segmentation refer to [AWESOME_VPS.md](https://github.com/GewelsJI/VPS/blob/main/docs/AWESOME_VPS.md).


## 2. Overview

### 2.1. Introduction

Colonoscopy is an effective technique for detecting colorectal polyps, which are highly related to colorectal cancer. 
In clinical practice, segmenting polyps from colonoscopy images is of great importance since it provides valuable 
information for diagnosis and surgery. However, accurate polyp segmentation is a challenging task, for two major reasons:
(i) the same type of polyps has a diversity of size, color and texture; and
(ii) the boundary between a polyp and its surrounding mucosa is not sharp. 

To address these challenges, we propose a parallel reverse attention network (PraNet) for accurate polyp segmentation in colonoscopy
images. Specifically, we first aggregate the features in high-level layers using a parallel partial decoder (PPD). 
Based on the combined feature, we then generate a global map as the initial guidance area for the following components. 
In addition, we mine the boundary cues using a reverse attention (RA) module, which is able to establish the relationship between
areas and boundary cues. Thanks to the recurrent cooperation mechanism between areas and boundaries, 
our PraNet is capable of calibrating any misaligned predictions, improving the segmentation accuracy. 

Quantitative and qualitative evaluations on five challenging datasets across six
metrics show that our PraNet improves the segmentation accuracy significantly, and presents a number of advantages in terms of generalizability,
and real-time segmentation efficiency (∼50fps).

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/framework-final-min.png"/> <br />
    <em> 
    Figure 1: Overview of the proposed PraNet, which consists of three reverse attention 
    modules with a parallel partial decoder connection. See § 2 in the paper for details.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/qualitative_results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX TITAN GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PraNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n PraNet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

1. Downloading necessary data:

    + obtain the testing datasets from their originating projects and move them into `./data/TestDataset/`. The expected datasets are CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, and Kvasir.
    
    + obtain the Kvasir-SEG and CVC-ClinicDB training datasets from their originating projects and move them into `./data/TrainDataset/`.
    
    + consult the [upstream PraNet repository](https://github.com/DengPingFan/PraNet) for current pretrained-weight availability and place an authorized artifact at `snapshots/PraNet_Res2Net/PraNet-19.pth`.
    
    + obtain Res2Net weights from the upstream Res2Net project or another authorized source.

1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.
    
    + Just enjoy it!

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!


### 3.2 Evaluating your trained model:

Matlab evaluation instructions are retained in `./eval/main.m`; run it to
generate evaluation results in `./res/`. Consult the
[upstream PraNet repository](https://github.com/DengPingFan/PraNet) for current
evaluation-toolbox availability.

Python: Please refer to the work of ACMMM2021 https://github.com/plemeri/UACANet


### 3.3 Pre-computed maps: 
Consult the [upstream PraNet repository](https://github.com/DengPingFan/PraNet)
for current pre-computed-map availability.


## 4. Citation

Please cite our paper if you find the work useful: 
    
    @inproceedings{fan2020pranet,
      title={Pranet: Parallel reverse attention network for polyp segmentation},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
      booktitle={International conference on medical image computing and computer-assisted intervention},
      pages={263--273},
      year={2020},
      organization={Springer}
    }

## 5. TODO LIST

> If you want to improve the usability or any piece of advice, please feel free to contact me directly ([E-mail](gepengai.ji@gmail.com)).

- [ ] Support `NVIDIA APEX` training.

- [ ] Support different backbones (
VGGNet, 
ResNet, 
[ResNeXt](https://github.com/facebookresearch/ResNeXt),
[iResNet](https://github.com/iduta/iresnet), 
and 
[ResNeSt](https://github.com/zhanghang1989/ResNeSt) 
etc.)

- [ ] Support distributed training.

- [ ] Support lightweight architecture and real-time inference, like MobileNet, SqueezeNet.

- [ ] Add more comprehensive competitors.

## 6. FAQ

1. If the image cannot be loaded in the page (mostly in the domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)

## 7. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---

**[⬆ back to top](#0-preface)**
