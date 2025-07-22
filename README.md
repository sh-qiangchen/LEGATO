# LEGATO

A PyTorch implementation of "**LEGATO: Learning to Forget Identity in Generative Models via Trajectory-consistent Neural ODE**". 



# Overview

**LEGATO** is a Neural ODE-based generative unlearning method to preserve generative capability while achieve the identity unlearning. **The core idea behind LEGATO is to model the unlearning process as a continuous trajectory**, reducing impact on model utility when unlearning specific identity.



# Getting Started

## Computing infrastructure

Most of our experiments were conducted on an NVIDIA GeForce RTX **3090** GPU, while a small portion of experiments that exceeded the memory capacity were performed on an NVIDIA **A100** GPU.



## Files to Run Code
|Filename|Description|
|-|-|
|ffhqrebalanced512-128.pkl|Pretrained weights of EG3D network.|
|w_avg_ffhqrebalanced512-128.pt|Average latent code computed from pretrained EG3D network.|
|model_ir_se50.pth|Pretrained weights of ArcFace network to compute identity loss.|
|encoder_FFHQ.pt|Pretrained weights of GOAEncoder for 3D GAN inversion.|
|CurricularFace_Backbone.pth|Pretrained weights of CurricularFace network to compute identity similarity.|
|ffhq_real_feat.npy|Statistics of Inception-v3 feature to easily compute ΔFID<sub>real</sub>.|

To run code, please download all of files via [GUIDE](https://github.com/KU-VGI/GUIDE).



# Erase an identity from pretrained 3D GAN
We provide sample data (./data/CelebAHQ/512) to run our code.  
```bash
# Random
python unlearn.py --exp guide --target extra --target_d 30.0 --local --adj --glob --seed 0

# FFHQ
python unlearn.py --exp guide --inversion goae --inversion_image_path ./data/FFHQ/00072.png --target extra --target_d 30.0 --local --adj --glob --seed 0

# CelebAHQ
python unlearn.py --exp legato --inversion goae --inversion_image_path ./data/CelebAHQ/512 --target extra --target_d 30.0 --local --adj --glob --seed 0
```

## Erase in the wild identities
If you want to erasure in the wild identities, please preprocess the images via [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch).  Otherwise, both the inversion encoder and pretrained generator can't recognize them correctly. 



# Evaluation
Evaluation on identity erasure:
```bash
python evaluate_id.py --exp legato
```
Evaluation on pretrained distribution preservation:
```bash
python evaluate_fid.py --exp legato
```
By running the above commands, we could obtain the results of:  
| |ID|ID<sub>others</sub>|FID<sub>pre</sub>|ΔFID<sub>real</sub>|
|-|-|-|-|-|
|GUIDE| 0.02 | 0.23                | 7.44              | 3.36                |
|LEGATO| 0.00 |0.18|6.09|1.78|

# Acknowledgement
Our code is based on [EG3D](https://github.com/NVlabs/eg3d), [GOAE](https://github.com/jiangyzy/GOAE), [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace) and [GUIDE](https://github.com/KU-VGI/GUIDE).