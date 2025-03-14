<div align="center">

# Effective Context Modeling Framework for Emotion Recognition in Conversations
</div>


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction
This is the official implementation of the paper *"Effective Context Modeling Framework for Emotion Recognition in Conversations"*. Our paper is published at **ICASSP 2025** 🎉.
<div align="center">
    <img src="static/framework.png" alt="Pipeline Overview" width="800"/>
    <p>Figure: Detailed architecture of <b>(A)</b> the proposed ConxGNN, <b>(B)</b> Inception Graph Block, and <b>(C)</b> HyperBlock.</p>
</div>

## Installation
Install the dependencies:
```bash
 conda env create -f environment/environment.yml
```
Read [`environment/helper.txt`](./environment/helper.txt) if some libraries can't be installed.

## Usage
To train the model, run the following command:
```bash
python train.py configs/meld.yaml       # for MELD
python train.py configs/iemocap6.yaml   # for IEMOCAP
```

## Acknowledgement
Part of the code is borrowed from the following repositories. We would like to thank the authors for their great work.
- [MultiEMO](https://github.com/TaoShi1998/MultiEMO)
- [M3Net](https://github.com/feiyuchen7/M3NET)
- [CORECT](https://github.com/leson502/CORECT_EMNLP2023)

## Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@INPROCEEDINGS{10888112,
  author={Van, Cuong Tran and Tran, Thanh V. T. and Nguyen, Van and Hy, Truong Son},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Effective Context Modeling Framework for Emotion Recognition in Conversations}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Emotion recognition;Attention mechanisms;Limiting;Speech recognition;Oral communication;Benchmark testing;Graph neural networks;Data models;Speech processing;Context modeling;Emotion Recognition in Conversations;Graph Neural Network;Hypergraph;Multimodal},
  doi={10.1109/ICASSP49660.2025.10888112}}
```
