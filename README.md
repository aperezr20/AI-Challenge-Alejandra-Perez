# Emotion Recognition and Worker Identification

![Project Logo](static/logo.png)

This project focuses on building a computer vision model for two key tasks:

1. **Emotion Detection Model:** Our Emotion Detection model is based on MobileNet V2, which is pre-trained on ImageNet and fine-tuned on a diverse dataset of [Natural Human Face Images for Emotion Recognition](https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition) from Kaggle. The classes that I considered for this problem were: angry, happy, neutral, and sad.

2. **Worker Identification:** For worker data collection, I sourced images from [Kaggle's Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition), which includes images from 19 different individuals. I considered 15 individuals from this source and collected data myself to obtain a 17-workers database. To achieve worker identification, I utilized face feature embeddings obtained through a ResNet model pre-trained on VGGFace. During inference, the model identifies workers by finding the closest match in the database based on face embeddings.

The code for MobileNet V2 training, face embedding dataset collection code, and the model server that runs on the cloud are in the /utils folder

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Contact](#contact)

## Features
- Worker recognition and emotion detection.
- Cloud deployment for easy accessibility.
- Web-based user interface for making inferences.

## Getting Started
Follow these instructions to set up and run the project on your local machine.

### Prerequisites
- Python 3.x
- Flask
- Pillow
- requests

### Installation 
1. Install the required Python packages:
```
pip install -r requirements.txt

```
2. Run the Flask app:
```
python app.py

```
### Usage
- Visit the web interface to upload an image. You may try with an image of yourself or with the ones in /images folder.
- The system will recognize the identity of the worker and predict their emotion based on the uploaded image.



  
