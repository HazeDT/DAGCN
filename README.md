# Domain Adversarial Graph Convolutional network (DAGCN)
This code is about the implementation of [Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions](https://ieeexplore.ieee.org/document/9410617).

![DAGCN](https://github.com/HazeDT/DAGCN/blob/main/Figure3.tif)

# Note
The DAGCN consists of a CNN and a [MRF_GCN](https://ieeexplore.ieee.org/document/9280401), and the framework of this code is based on [Unsupervised Deep Transfer Learning for Intelligent Fault Diagnosis: An Open Source and Comparative Study](https://arxiv.org/abs/1912.12528v1).


# Implementation
python ./DAGCN/train_advanced.py --model_name DAGCN_features  --checkpoint_dir ./DAGCN/results/   --data_name CWRU --data_dir D:/Data/西储大学轴承数据中心网站 --transfer_task [3],[0]  --last_batch True 


# Citation
MRF_GCN: 
@ARTICLE{MRF_GCN,
  author={T. {Li} and Z. {Zhao} and C. {Sun} and R. {Yan} and X. {Chen}},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Multi-receptive Field Graph Convolutional Networks for Machine Fault Diagnosis}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIE.2020.3040669}}

DAGCN:
@ARTICLE{9410617,
  author={T. {Li} and Z. {Zhao} and C. {Sun} and R. {Yan} and X. {Chen}},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIM.2021.3075016}}



