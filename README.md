# Real-Time Crime Detection using Swin Transformer

This project implements an advanced **real-time crime detection system** using the **Swin Transformer** architecture. The model is trained on video data to detect criminal activities in real-time, utilizing a trimmed version of the **UCF Crime Detection Dataset**.

## Project Overview

The objective of this project is to enhance crime detection accuracy and reduce latency in real-time surveillance systems. The Swin Transformer model has been employed due to its state-of-the-art performance on video understanding tasks, providing both high accuracy and efficiency.

### Key Features:
- **Model:** Swin Transformer (Shifted Window Transformer)
- **Dataset:** UCF Crime Detection (trimmed version)
- **Accuracy:** 94% validation accuracy
- **F1-score:** 0.92 for detecting criminal activities
- **Latency:** Real-time detection with latency under 200ms

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Crime detection in surveillance footage is a critical task for ensuring public safety. Traditional methods rely on manual monitoring, which is time-consuming and inefficient. This project uses **Swin Transformer**, a hierarchical vision transformer model, to automate the detection of criminal activities from video feeds in real time.

## Dataset

The **UCF Crime Detection Dataset** contains over 1,000 video clips, including both normal and crime-related activities. For this project:
- We used a **trimmed version** of the dataset to focus on the most relevant sections for crime detection.
- The videos were preprocessed for normalization and temporal consistency.

## Model Architecture

The **Swin Transformer** is a hierarchical vision transformer that shifts windows to capture both local and global context from the input video frames. It provides:
- **Local attention** within non-overlapping windows.
- **Global attention** by shifting windows in consecutive layers, allowing efficient video analysis without sacrificing accuracy.

Key model parameters:
- **Backbone:** Swin Transformer Tiny
- **Input Size:** 224x224 pixel frames
- **Optimizer:** AdamW
- **Loss Function:** Cross-Entropy Loss

## Preprocessing

To prepare the dataset for the Swin Transformer model, the following preprocessing steps were applied:
1. **Normalization**: Scaling pixel values to the range [0, 1].
2. **Temporal Subsampling**: Reducing the frame rate to focus on key segments.
3. **Data Augmentation**: Techniques such as random cropping, flipping, and rotation were used to increase the robustness of the model.

## Training

The Swin Transformer model was trained over **500 epochs** with the following configuration:
- **Batch Size:** 32
- **Optimizer:** AdamW
- **Learning Rate:** 0.0001 with a cosine decay schedule
- **Loss Function:** Cross-Entropy Loss

During training, the model learned to classify video frames as either normal or depicting criminal activities. The training process involved intensive computation to optimize for both accuracy and latency.

## Evaluation

The model was evaluated using standard metrics like **accuracy**, **precision**, **recall**, and **F1-score**. The Swin Transformer achieved the following results:
- **Validation Accuracy:** 94%
- **F1-Score:** 0.92
- **Precision/Recall:** High across all classes

## Results

The Swin Transformer model delivered superior performance in real-time crime detection:
- **Real-time detection** with a latency of under 200ms.
- High accuracy and robustness in identifying various criminal activities from video feeds.

## Usage

- Clone this repository
- Go to the main.py file and replace the existing path with the weights given [here](https://drive.google.com/drive/folders/1kJmc1ogtt1jD5x9XkaKU3ivSU32f3tll?usp=drive_link)
- Now run the main.py which opens the local machine camera for detection
  

