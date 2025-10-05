# Chest X-Ray Classification Using CNN  

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://pneumonia-app-detector.streamlit.app/)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Project Overview  

This project is part of the **Machine Learning Capstone (ML-2)** and demonstrates the use of **Convolutional Neural Networks (CNNs)** for medical image classification.  
The model learns to classify **chest X-ray images** as **Normal** or **Pneumonia**, automating an essential diagnostic process for respiratory diseases.  

🔗 **Live App:** [Pneumonia Detector Streamlit App](https://pneumonia-app-detector.streamlit.app/)  

---

## Problem Statement  


Pneumonia is an infection that inflames the air sacs (alveoli) in one or both lungs. 
When you have pneumonia, these air sacs fill up with fluid or pus, which can make it hard to breathe and limit the amount of oxygen that gets into your bloodstream. 
Manual analysis of chest X-rays is time-consuming and prone to human error, especially in low-resource medical environments.  

> **Goal:** Build and train a Convolutional Neural Network (CNN) to automatically detect pneumonia from chest X-ray images, enabling faster and more accessible diagnosis.

---

## Objectives  

- Preprocess and augment chest X-ray images for training.  
- Build a deep learning model using CNN architecture.  
- Train and evaluate the model for reliable classification.  
- Deploy the trained model as a **Streamlit web app** for live predictions.  
- Visualize performance metrics such as loss, accuracy, and confusion matrix.  

---

## Dataset  

**Source:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Structure:**
```css
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Each subfolder contains labeled X-ray images used for training, validation, and testing.

---

## Model Architecture  

The CNN model includes:  
- Convolutional + MaxPooling layers for feature extraction  
- Dropout layers for regularization  
- Dense layers for classification  
- Softmax output for binary decision  

Framework: **TensorFlow / Keras**

---

## Technologies Used  

| Category | Tools & Libraries |
|-----------|------------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV, PIL |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | tqdm, os, random, sklearn |
| **Deployment** | Streamlit |

---

## Setup & Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/isaiahokumu/ML-2-Capstone.git
cd ML-2-Capstone
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the Chest X-Ray (Pneumonia) dataset and place it in a folder named chest_xray/ inside the project directory.

### 4. Run the Notebook
```bash
jupyter notebook main.ipynb
```

### 5. Launch the Deployed App
Visit the deployed model here:
[![Pneumonia-Detector-App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://pneumonia-app-detector.streamlit.app/)  

### 6. Model Evaluation

| Metric                 | Description                                 |
| ---------------------- | ------------------------------------------- |
| **Accuracy**           | Overall classification correctness          |
| **Precision / Recall** | False positive & false negative control     |
| **F1-Score**           | Balance between precision and recall        |
| **Confusion Matrix**   | Visualization of classification performance |

```yaml
Training Accuracy: 92%
Validation Accuracy: 89%
Test Accuracy: 90%
```


## Visualizations

Sample chest X-ray images

Training vs Validation Accuracy/Loss plots

Confusion Matrix Heatmap

Feature maps and activation layers (optional)


## Future Work

Implement Transfer Learning using pretrained models (VGG16, ResNet50).

Integrate Grad-CAM for visual model explainability.

Deploy model as a Flask / FastAPI REST API for scalability.

Expand the dataset for multi-class classification (Viral vs Bacterial Pneumonia).

Integrate into hospital diagnostic workflows.

 
 ## Author

Isaiah Okumu
📍 Data Science & Machine Learning Enthusiast

## License
````sql
MIT License

Copyright (c) 2025 Isaiah Okumu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction...
````
