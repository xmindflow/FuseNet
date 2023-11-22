# FuseNet: Self-Supervised Dual-Path Network for Medical Image Segmentation 

[![arXiv](https://img.shields.io/badge/arXiv-2308.13442-b31b1b.svg)]() [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mindflow-institue/FuseNet/blob/main/FuseNet_colab.ipynb)


Semantic segmentation, a crucial task in computer vision, often relies on labor-intensive and costly annotated datasets for training. In response to this challenge, we introduce FuseNet, a dual-stream framework for self-supervised semantic segmentation that eliminates the need for manual annotation. FuseNet leverages the shared semantic dependencies between the original and augmented images to create a clustering space, effectively assigning pixels to semantically related clusters, and ultimately generating the segmentation map. Additionally, FuseNet incorporates a cross-modal fusion technique that extends the principles of CLIP by replacing textual data with augmented images. This approach enables the model to learn complex visual representations, enhancing robustness against variations similar to CLIP’s text invariance. To further improve edge alignment and spatial consistency between neighboring pixels, we introduce an edge refinement loss. This loss function considers edge information to enhance spatial coherence, facilitating the grouping of nearby pixels with similar visual features. Extensive experiments on skin lesion and lung segmentation datasets demonstrate the effectiveness of our method.

<br>

![FuseNet](https://github.com/xmindflow/FuseNet/assets/61879630/5da3daef-24f7-4dae-b702-1ba689799b0b)
<br>



## Updates
- If you found this paper useful, please consider checking out our previously accepted papers at MIDL and ICCV:
`MS-Former` [[Paper](https://openreview.net/forum?id=pp2raGSU3Wx)] [[GitHub](https://github.com/mindflow-institue/MS-Former)], and `S3-Net` [[Paper](https://openreview.net/forum?id=pp2raGSU3Wx)] [[GitHub](https://github.com/mindflow-institue/MS-Former)] ♥️✌🏻

- November 22, 2023: First release of the code.

## Installation

```bash
pip install -r requirements.txt
```

## Run Demo
Put your input images in the ```input_images/image``` folder and just simply run the ```FuseNet.ipynb``` notebook ;)

## Experiments

<p align="center">
  <img src="https://github.com/xmindflow/FuseNet/assets/61879630/e2216fae-bb73-4dd8-8191-b8300b876d21" width="800">
</p>


## Citation
If this code helps with your research, please consider citing the following paper:
</br>

```python
@inproceedings{
}
```
