
##  LASDT

A PyTorch implementation for 'Local Auxiliary Spatial-Spectral Decoupling Transformer Network for Cross-Scene Hyperspectral Image Classification'
Qiusheng Chen, Zhuoqun Fang, Zhaokui Li, Qian Du, Shizhuo Deng, Tong Jia and Dongyue Chen*

## 📦 Dependencies

The code requires the following packages (tested with Python 3.10+):

```bash
torch==1.13
numpy==1.26.4
einops==0.7.0
spectral==0.23.1
imageio==2.37.0
scipy==1.15.2
scikit-learn==1.2.1
matplotlib==3.7.0
```
## 🗂 Dataset
You can download the required dataset from https://doi.org/10.6084/m9.figshare.26761831. 
You can modify the dataset storage directory in lines 35-55 of main_train.py and lines 26-46 of main_eval.py accordingly.
## 📝 Pretrain Model
Our pretrained model weight files can be downloaded from https://pan.baidu.com/s/1PpPGkOpf_QLGuuMNkoowrA?pwd=pisx


## 🚀 How to Run

```bash
# Training using Houston dataset with default params
# Change the directory to the lasdt folder.
cd /root/data/Projects/lasdt
./run_houston_train.sh

# Testing using Houston dataset with default params
./run_eval.sh
```