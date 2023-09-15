# BS-Loss
This repository is an official PyTorch implementation of the paper "Boundary-Sensitive Loss Function With Location Constraint for Hard Region Segmentation" [paper] (https://ieeexplore.ieee.org/document/9950613) from IEEE Journal of Biomedical and Health Informatics (JBHI) 2022.
## Requirements (PyTorch)
Core implementation (to integrate the bs loss into your own code):

- python 3.8
- pytorch 1.8.1
- opencv-python 4.6.0.66
- numpy

To reproduce our experiments:

- python 3.8
- pytorch 1.8.1
- copy
- numpy
- matplotlib
- scikit-image
- random
- glob
- tqdm


## Usage
The implementation of BS loss has three key functions:

- the BS loss itself (`boundary_sensitive_loss` in [BS_loss.py#L67](https://github.com/dujie-szu/BS-Loss/blob/main/BS_loss.py#:~:text=(prediction%2C-,label%2C%20alpha,-%2C%20eps%3D));
- the dilation and erosion operations (`get_boundary` in [BS_loss.py#L40](https://github.com/dujie-szu/BS-Loss/blob/main/BS_loss.py#:~:text=GT%20and%20Pred-,def%20get_boundary,-(img)%3A));
- the Location Constraint (`location_constraint` in [BS_loss.py#L25](https://github.com/dujie-szu/BS-Loss/blob/main/BS_loss.py#:~:text=def%20location_constraint(-,prediction%2C,-label)%3A));

If you want to use BS loss as the loss function only, you can directly use the `BSLoss` in [BS_loss.py#L6](https://github.com/dujie-szu/BS-Loss/blob/main/BS_loss.py#:~:text=class%20BSLoss-,(_Loss,-)%3A)). There is an optional argument in the `BSLoss`, but we suggest to use the default setting of `alpha=0.8`. If you want to use BS loss with Location Constraint, you can directly use the `BSL_LC` in [BS_loss.py#L15](https://github.com/dujie-szu/BS-Loss/blob/main/BS_loss.py#:~:text=class-,BSL_LC(_Loss,-)%3A)).

It should be noted that our method is designed for binary segmentation tasks, and you need to ensure that the shape of your network input and output is *[N,1,H,W]* or *[N,H,W]*.

## Cite BS-loss
---------
If you used BS loss in your research projects, please remember to cite our reference paper published at the [IEEE Journal of Biomedical and Health Informatics (JBHI) 2022](https://ieeexplore.ieee.org/document/9950613/). This will help us make BS loss known in the machine learning community, ultimately making a better tool for everyone:
```
@ARTICLE{9950613,  
author={Du, Jie and Guan, Kai and Liu, Peng and Li, Yuanman and Wang, Tianfu},  journal={IEEE Journal of Biomedical and Health Informatics},   
title={Boundary-Sensitive Loss Function with Location Constraint for Hard Region Segmentation},   
year={2022},  
volume={},  
number={},  
pages={1-12},  
doi={10.1109/JBHI.2022.3222390}}
```
