# Image Transformation As Diffusion Visual Programmers
------

Our [arxiv](https://arxiv.org/abs/2401.09742) version is currently available. Please check it out! 🔥🔥🔥

This repository contains the official PyTorch implementation for Image Transformation As Diffusion Visual Programmers. Our work is built upon [VisProg](https://github.com/allenai/visprog), [Null-Text Inversion](https://null-text-inversion.github.io/) and other repos under Stable Diffusion framework. We thank the great work of them. 

## Abstract
We introduce the novel Diffusion Visual Programmer (DVP), a neuro-symbolic image translation framework. Our proposed DVP seamlessly embeds a conditionflexible diffusion model within the GPT architecture, orchestrating a coherent sequence of visual programs (i.e., computer vision models) for various pro-symbolic steps, which span RoI identification, style transfer, and position manipulation, facilitating transparent and controllable image translation processes. Extensive experiments demonstrate DVP’s remarkable performance, surpassing concurrent arts. This success can be attributed to several key features of DVP: First, DVP achieves condition-flexible translation via instance normalization, enabling the model to eliminate sensitivity caused by the manual guidance and optimally focus on textual descriptions for high-quality content generation. Second, the framework enhances in-context reasoning by deciphering intricate high-dimensional concepts in feature spaces into more accessible low-dimensional symbols (e.g., [Prompt], [RoI object]), allowing for localized, context-free editing while maintaining overall coherence. Last but not least, DVP improves systemic controllability and explainability by offering explicit symbolic representations at each programming stage, empowering users to intuitively interpret and modify results. Our research marks a substantial step towards harmonizing artificial image translation processes with cognitive intelligence, promising broader applications.

## Installation
This code was tested with Python 3.8, [Pytorch](https://pytorch.org/) 1.11 using pre-trained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over [VisProg](https://github.com/allenai/visprog), [Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256) and  [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4).

Below are quick steps for installation:

```shell
conda create -n dvp python=3.8 pytorch=1.11 cudatoolkit=11.3 torchvision==0.12.0 -c pytorch -y
conda activate dvp
pip3 install -e .
```

Please refer to [VisProg](https://github.com/allenai/visprog) for more detailed installation and dependencies.


## Citation

If you find our work helpful in your research, please cite it as:

```
@article{han2024image,
  title={Image Translation as Diffusion Visual Programmers},
  author={Han, Cheng and Liang, James C and Wang, Qifan and Rabbani, Majid and Dianat, Sohail and Rao, Raghuveer and Wu, Ying Nian and Liu, Dongfang},
  journal={ICLR},
  year={2024}
}
```