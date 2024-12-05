# Machine Learning 24 homework repo
This repo hosts some classic classification model, we adopt it for few-shot learning and test the performance.
- kNN
- SVM
- MLP
- TCN

## Getting Started 

### Prerequisites
0. Clone this repo

```
git clone https://github.com/XuanCheg/ML24-Proj1.git
cd ML24-Proj1
```

1. Install dependencies.

This code requires Python 3.10, PyTorch, and a few other Python libraries. 
We recommend creating conda environment and installing all the dependencies as follows:
```bash
# create conda env
conda create --name ml24 python=3.10
# activate env
conda actiavte ml24
# install pytorch with CUDA 12.1
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# install other python packages
pip install tqdm ipython easydict tensorboard tabulate scikit-learn
```
The PyTorch version we tested is `2.4.0`.

2. Spilt datasets
Your can spilt the `.mat` datasets easily by running code as follows:
```bash
cd data
python spilt.py
```
It will convert the datasets into `.npz` format and spilt them.
### Training

Training can be launched by running the following command:
```
bash ./train.sh 
```
This will train TCN for 200 epochs on the OCD_90_200_fMRI train split, The training is very fast, it can be done within 5 min using a single RTX 3070Ti GPU. The checkpoints and other experiment log files will be written into `results`. For training under different settings, you can append additional command line flags to the command above or change `train.sh`.

For more configurable options, please checkout our config file [utils/config.py](utils/config.py).

Train log with performance on val and test will be written into `results` after training
