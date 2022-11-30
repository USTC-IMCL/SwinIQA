# SwinIQA
SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment
> [**SwinIQA**](https://arxiv.org/abs/2205.04264), Jianzhao Liu, Xin Li, Yanding Peng, Tao Yu, Zhibo Chen,        
> Won **the first prize** in the competition of **5th Workshop and Challenge on Learned Image Compression (CLIC2022)** on the **Perceptual metrics Track**.

> Accepted by CVPR2022 Workshop 
![image](https://github.com/USTC-IMCL/SwinIQA/blob/main/fig/SwinIQA.png)

## Abstract
Image compression has raised widespread concern recently due to its significant importance for multimedia storage and transmission. Meanwhile, a reliable image quality assessment (IQA) for compressed images can not only help to verify the performance of various compression algorithms but also help to guide the compression optimization in turn. In this paper, we design a full-reference image quality assessment metric SwinIQA to measure the perceptual quality of compressed images in a learned Swin distance space. It is known that the compression artifacts are usually non-uniformly distributed with diverse distortion types and degrees. To warp the compressed images into the shared representation space while maintaining the complex distortion information, we extract the hierarchical feature representations from each of the stage of the Swin Transformer. Besides, we utilize cross attention operation to map the extracted feature representations into a learned Swin distance space. Experimental results show that the proposed metric achieves higher consistency with human's perceptual judgment comapred with both traditional methods and learning-based methods on CLIC dataset.

## Usages
Before the testing, the file '/data/xx.csv' is required to be edited to storage the file list you want to test as:
1) Reference, A, B
2) Reference, A
Here, A and B are the coded images with different codecs.

When you require to compare the quality of two distorted images given the referenced image, you can run the test code with:
```
 python clic_test.py --compare=True
```

If you only require to get the quality of the distorted image compared with the referenced one, you can run the test code with:
```
python clic_test.py --compare=False
```

The pretrained weights are released [Model] (https://drive.google.com/file/d/1QtOnb4HjU-k_CCmpAACYNDrQZ0Me9y22/view?usp=sharing)

The pretrained model requires to be moved to the folder "weights" 

## Cite US
Please cite us if this work is helpful to you.
```
@inproceedings{liu2022swiniqa,
  title={SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment},
  author={Liu, Jianzhao and Li, Xin and Peng, Yanding and Yu, Tao and Chen, Zhibo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1795--1799},
  year={2022}
}
```
