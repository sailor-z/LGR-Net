# LGR-Net: Rotation Invariant Point Cloud Classification: Where Local Geometry Meets Global Topology
***
This repository is a reference implementation for "Rotation invariant point cloud analysis: Where local geometry meets global topology". If you use this code in your research, please cite the paper.

```
@article{zhao2019rotation,
  title={Rotation Invariant Point Cloud Classification: Where Local Geometry Meets Global Topology},
  author={Zhao, Chen and Yang, Jiaqi and Xiong, Xin and Zhu, Angfan and Cao, Zhiguo and Li, Xin},
  journal={arXiv preprint arXiv:1911.00195},
  year={2019}
}
```
# Train model on ModelNet40
***
The code of svd is borrowed from [torch-batch-svd] (https://github.com/KinglittleQ/torch-batch-svd). Please installing it before runing the training code. 
```
cd ./pt_utils/svd
python setup.py install
```

```
cd ../../
python ./train_cls.py
```
