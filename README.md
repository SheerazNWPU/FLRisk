# FLRisk- Risk-Aware Adaptive Federated Learning for Breast Cancer Biomarker Prediction

This project contains the code for our work titled **"FLRisk- Risk-Aware Adaptive Federated Learning for Breast Cancer Biomarker Prediction."** This code enables the detection of mispredictions in any federated learning task trained using any model. In our work, we focus on multilabel biomarker prediction for breast cancer in a federated environment. Our general experiments use ResNet50 as the baseline model, which can be replaced with any deep neural network (DNN) model, including Transformers and Graph Neural Networks.


## Overall Framework
The overall framework of our work is shown below:

![Risk Aware Adaptive Federated Learning](Adaptive%20Federated%20Learning.png)


## Installation
Install the required packages listed in `Requirements.txt`.

## Usage
```bash
PrepareRiskData
OneSidedRules
Common
python Main.py
