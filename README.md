**Continuous Tensor Toolbox (Python)**

Abstract: Since higher-order tensors are naturally suitable for representing multi-dimensional data in real-world, e.g., color images and videos, low-rank tensor representation has become one of the emerging areas in machine learning and computer vision. However, classical low-rank tensor representations can solely represent multi-dimensional discrete data on meshgrid, which hinders their potential applicability in many scenarios beyond meshgrid. To break this barrier, we propose a low-rank tensor function representation (LRTFR) parameterized by multilayer perceptrons (MLPs), which can continuously represent data beyond meshgrid with powerful representation abilities. Specifically, the suggested tensor function, which maps an arbitrary coordinate to the corresponding value, can continuously represent data in an infinite real space. Parallel to discrete tensors, we develop two fundamental concepts for tensor functions, i.e., the tensor function rank and low-rank tensor function factorization, and utilize MLPs to paramterize factor functions of the tensor function factorization. We theoretically justify that both low-rank and smooth regularizations are harmoniously unified in LRTFR, which leads to high effectiveness and efficiency for data continuous representation. Extensive multi-dimensional data recovery applications arising from image processing (image inpainting and denoising), machine learning (hyperparameter optimization), and computer graphics (point cloud upsampling) substantiate the superiority and versatility of our method as compared with state-of-the-art methods. Especially, the experiments beyond the original meshgrid resolution (hyperparameter optimization) or even beyond meshgrid (point cloud upsampling) validate the favorable performances of our method for continuous representation.

**Citation**

@ARTICLE{LRTFR,
author={Luo, Yisi and Zhao, Xile and Li, Zhemin and Ng, Michael K. and Meng, Deyu},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Low-Rank Tensor Function Representation for Multi-Dimensional Data Recovery}, 
year={2023},\\
doi={10.1109/TPAMI.2023.3341688}}
