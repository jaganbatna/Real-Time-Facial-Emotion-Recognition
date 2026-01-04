# 🎭 Real-Time Facial Emotion Recognition System (CNN)

A real-time facial emotion recognition system built using **Deep Learning and Computer Vision**, capable of accurately classifying human emotions from live webcam feeds and static images while running efficiently on standard CPU hardware.

---

## 🚀 Project Overview

This project demonstrates an end-to-end **machine learning deployment pipeline**—from data preprocessing and CNN model training to real-time inference and GUI-based user interaction.

The system detects faces, preprocesses facial regions, and classifies expressions into seven core emotions using a lightweight Convolutional Neural Network trained on the **FER-2013 dataset**.

The focus is on **real-time performance, robustness, and deployability**, not just model accuracy.

---

## 🎯 Key Features

- Real-time emotion recognition via webcam  
- Emotion prediction from uploaded images  
- Face detection using Haarcascade  
- CNN-based emotion classification  
- GUI-based interaction using Tkinter  
- Optimized for CPU execution (no GPU required)  

---

## 😄 Emotions Classified

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 🧠 System Workflow

Input Image / Webcam
→ Grayscale Conversion
→ Face Detection (Haarcascade)
→ Face Cropping
→ Resize to 48×48
→ Normalization
→ CNN Inference
→ Softmax Probability Output
→ Emotion Label + Confidence Display


---

## 🏗️ Model Architecture

- Conv2D (32 filters) + ReLU  
- MaxPooling  
- Conv2D (64 filters) + ReLU  
- MaxPooling  
- Flatten  
- Dense (128 units) + ReLU  
- Dropout (0.3)  
- Dense (7 units) + Softmax  

**Training Configuration**
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 20  
- Batch Size: 64  

---

## 📊 Performance Summary

- Stable training and validation accuracy  
- Strong recognition of dominant emotions (Happy, Neutral)  
- Expected confusion between visually similar emotions  
- Good generalization on unseen real-world images  

---

## 🖥️ GUI & Real-Time Inference

- Built using Tkinter  
- Supports image upload and live webcam detection  
- Displays bounding boxes, predicted emotion, and confidence score  
- Smooth real-time performance on standard laptops  

---

## 🧪 Dataset

**FER-2013**
- 35,887 grayscale facial images  
- Resolution: 48×48  
- Real-world variations in lighting, pose, and expressions  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pandas  
- Tkinter  

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/facial-emotion-recognition-cnn.git
cd facial-emotion-recognition-cnn
```
2.Install required libraries
```
pip install tensorflow opencv-python numpy pandas pillow
```

3.Run the application

🔮 Future Enhancements

Replace Haarcascade with MTCNN or YOLO-based face detection

Improve class imbalance handling using weighted loss or data augmentation

Deploy on mobile and edge devices using TensorFlow Lite

Extend to multimodal emotion recognition (audio + text)
