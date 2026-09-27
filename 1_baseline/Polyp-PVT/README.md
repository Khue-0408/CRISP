# Polyp-PVT

by Bo Dong, Wenhai Wang, Jinpeng Li, Deng-Ping Fan.

This repo is the official implementation of ["Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers"](https://arxiv.org/pdf/2108.06932.pdf). 


<img src="./Figs/visual.gif" width="100%" />



## 1. Introduction
**Polyp-PVT** is initially described in [arxiv](https://arxiv.org/pdf/2108.06932.pdf).


Most polyp segmentation methods use CNNs as their backbone, leading to two key issues when exchanging information between the encoder and decoder: 1) taking into account the differences in contribution between different-level features; and 2) designing effective mechanism for fusing these features.
Different from existing CNN-based methods, we adopt a transformer encoder, which learns more powerful and robust representations. 
In addition, considering the image acquisition influence and elusive properties of polyps, we introduce three novel modules, including a cascaded fusion module (CFM), a camouflage identification module (CIM), and a similarity aggregation module (SAM).
Among these, the CFM is used to collect the semantic and location information of polyps from high-level features, while the CIM is applied to capture polyp information disguised in low-level features. 
With the help of the SAM, we extend the pixel features of the polyp area with high-level semantic position information to the entire polyp area, thereby effectively fusing cross-level features.
The proposed model, named **Polyp-PVT** , effectively suppresses noises in the features and significantly improves their expressive capabilities. 


Polyp-PVT achieves strong performance on image-level polyp segmentation (`0.808 mean Dice` and `0.727 mean IoU` on ColonDB) and
video polyp segmentation (`0.880 mean dice` and `0.802 mean IoU` on CVC-300-TV), surpassing previous models by a large margin.



## 2. Framework Overview
![](https://github.com/DengPingFan/Polyp-PVT/blob/main/Figs/network.png)


## 3. Results
### 3.1 Image-level Polyp Segmentation
![](https://github.com/DengPingFan/Polyp-PVT/blob/main/Figs/github_r1.png)

### 3.2 Image-level Polyp Segmentation Compared Results:
Historical result artifacts were distributed by the upstream project. Consult
the [Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT) for current
availability.

### 3.3 Video Polyp Segmentation
![](https://github.com/DengPingFan/Polyp-PVT/blob/main/Figs/github_r2.png)

### 3.4 Video Polyp Segmentation Compared Results:
Historical video result artifacts were distributed by the upstream project.
Consult the [Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT) for
current availability.

## 4. Usage:
### 4.1 Recommended environment:
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```
### 4.2 Data preparation:
Obtain the training and testing datasets from their originating projects and
move them into `./dataset/`.


### 4.3 Pretrained model:
Consult the [Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT)
for current pretrained-model availability, then place an authorized artifact
in `./pretrained_pth/` for initialization.

### 4.4 Training:
Clone the repository:
```
git clone https://github.com/DengPingFan/Polyp-PVT.git
cd Polyp-PVT 
bash train.sh
```

### 4.5 Testing:
```
cd Polyp-PVT 
bash test.sh
```

### 4.6 Evaluating your trained model:

Matlab: Please refer to the work of MICCAI2020 ([link](https://github.com/DengPingFan/PraNet)).

Python: Please refer to the work of ACMMM2021 ([link](https://github.com/plemeri/UACANet)).

Please note that we use the Matlab version to evaluate in our paper.


### 4.7 Well trained model:
Consult the [Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT)
for current trained-model availability and place an authorized artifact in
`./model_pth/`.

### 4.8 Pre-computed maps:
Consult the [Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT)
for current pre-computed-map availability.



## 5. Citation:
```
@aticle{dong2023PolypPVT,
  title={Polyp-PVT: Polyp Segmentation with PyramidVision Transformers},
  author={Bo, Dong and Wenhai, Wang and Deng-Ping, Fan and Jinpeng, Li and Huazhu, Fu and Ling, Shao},
  journal={CAAI AIR},
  year={2023}
}
```

## 6. Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [EAGRNet](https://github.com/tegusi/EAGRNet) and [MSEG](https://github.com/james128333/HarDNet-MSEG), which have provided the basis for our framework.

## 7. FAQ:
If you want to improve the usability or any piece of advice, please feel free to contact me directly (bodong.cv@gmail.com).

## 8. License
The source code is free for research and education use only. Any comercial use should get formal permission first.
