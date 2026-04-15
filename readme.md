# Pneumonia Detection using ResNet18

## Overview
This project focuses on fine-tuning a pretrained ResNet18 model to classify chest X-ray images into two categories: Normal and Pneumonia.

## Problem Statement
Accurate detection of pneumonia from X-ray images is critical, as missing a pneumonia case (false negative) can have serious medical consequences.

## Approach
- Used a pretrained ResNet18 model (trained on ImageNet)
- Applied transfer learning:
  - Froze early layers to preserve general features
  - Replaced the final fully connected layer for binary classification
- Implemented a manual training loop in PyTorch (no high-level wrappers)

## Dataset
- Chest X-ray images divided into:
  - Normal
  - Pneumonia
- Data is loaded using PyTorch DataLoader with appropriate transformations

## Training
- Custom training loop:
  - Forward pass
  - Loss computation
  - Backward propagation
  - Optimizer step
- Loss Function: CrossEntropyLoss
- Optimizer: (to be defined)

## Evaluation
- Evaluated model on validation data
- Metrics used:
  - Accuracy
  - Recall (important to minimize missed pneumonia cases)

## Project Structure