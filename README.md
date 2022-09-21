# SwinIQA
SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment
![image](https://github.com/USTC-IMCL/SwinIQA/tree/main/fig/SwinIQA.png)
> [**SwinIQA**](https://arxiv.org/abs/2205.04264), Jianzhao Liu, Xin Li, Yanding Peng, Tao Yu, Zhibo Chen,        
> Won **the first prize** in the competition of **5th Workshop and Challenge on Learned Image Compression (CLIC2022)** on the **Perceptual metrics Track**.

> Accepted by CVPR2022 Workshop 

## Abstract
Image compression has raised widespread concern recently due to its significant importance for multimedia storage and transmission. Meanwhile, a reliable image quality assessment (IQA) for compressed images can not only help to verify the performance of various compression algorithms but also help to guide the compression optimization in turn. In this paper, we design a full-reference image quality assessment metric SwinIQA to measure the perceptual quality of compressed images in a learned Swin distance space. It is known that the compression artifacts are usually non-uniformly distributed with diverse distortion types and degrees. To warp the compressed images into the shared representation space while maintaining the complex distortion information, we extract the hierarchical feature representations from each of the stage of the Swin Transformer. Besides, we utilize cross attention operation to map the extracted feature representations into a learned Swin distance space. Experimental results show that the proposed metric achieves higher consistency with human's perceptual judgment comapred with both traditional methods and learning-based methods on CLIC dataset.


