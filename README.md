# Lane2Seq: Unified Lane Detection with Transformers

![Project Workflow](https://i.imgur.com/G5gqB44.png)

## 📖 Introduction
[cite_start]This project is an implementation of **Lane2Seq: Towards Unified Lane Detection via Sequence Generation**, a novel framework for lane detection. [cite_start]Instead of relying on traditional, task-specific approaches like segmentation or curve fitting, Lane2Seq unifies these methods by casting lane detection as a sequence generation task. [cite_start]This approach avoids complex, task-specific head networks and loss functions typical of older methods.

## ✨ Features
* [cite_start]**Unified Architecture**: A single transformer-based encoder-decoder model handles multiple lane representations.
* [cite_start]**Sequence Generation**: Lane detection is framed as a sequence-to-sequence task, where the model generates a series of tokens that describe a lane's geometry.
* [cite_start]**ViT Encoder**: A Vision Transformer (ViT) encoder, pre-trained using Masked Autoencoders (MAE), is used to learn powerful image features.
* [cite_start]**Format Flexibility**: Supports three distinct lane representations: **Segmentation**, **Anchor**, and **Parameter** formats, controlled by a format-specific input prompt.
* [cite_start]**Reproducible Pipeline**: Includes a complete training, inference, and evaluation pipeline for the TuSimple dataset[cite: 1, 2].

## ⚙️ Getting Started

### Prerequisites
Make sure you have a Python environment set up with PyTorch. You can install the required packages using `pip`:
```bash
pip install torch torchvision numpy pyyaml tqdm transformers safetensors opencv-python shapely Pillow
