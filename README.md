# Automatic Data Augmentation for Medical Image Segmentation Using Adaptive Sequence-Length Based Deep Reinforcement Learning

This repository is the official implementation of CIBM paper [Automatic Data Augmentation for Medical Image Segmentation Using Adaptive Sequence-Length Based Deep Reinforcement Learning]. 

## Usage
### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
### Datasets
To download datasets:
- [Cardiac](http://www.cardiacatlas.org)
- [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- [LiTs](http://medicaldecathlon.com/)

### Training

To train the U-Net model, run this command:

```train
python main_base.py
```

To train the DQN model, run this command:

```train
python main_dqn.py
```

### Evaluation

To evaluate the augmented dataset, run this command:
```eval
python main_base.py 
```
