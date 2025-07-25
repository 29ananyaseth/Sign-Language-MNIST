##  Sign Language MNIST

## 📌 Description

Sign Language MNIST is a drop-in replacement for the classic MNIST dataset, designed for hand gesture recognition. It contains grayscale 28x28 images of American Sign Language (ASL) alphabets (A–Y, excluding J and Z due to motion). The dataset is ideal for training and benchmarking machine learning models on multi-class image classification tasks.

---

## 📊 Dataset Format

Each row in the dataset represents a label (A–Y) and 784 pixel values:

```
label,pixel1,pixel2,…,pixel784
```

- **Image Size:** 28x28 grayscale
- **Train Samples:** 27,455
- **Test Samples:** 7,172

---

## 🧪 Image Processing & Augmentation

Original color images were:
- Cropped to hands-only
- Grayscaled and resized
- Augmented using:
  - Random filters and pixelation
  - ±15% brightness/contrast
  - ±3° rotation

---

## ✅ Accuracy

**Model Accuracy:** 96.37%

---

## 📉  Confusion Matrix

![Confusion Matrix]()

---

## 🖼️ Sample Prediction Screenshots

> Add your own screenshot below showing sample predictions during testing.

![Sample Screenshot](sample_prediction_placeholder.png)

---

## 💻 Technologies Used

### Core Technologies
- **Python**
- **Jupyter Notebook**

### Data Handling & Visualization
- `pandas`
- `numpy`
- `matplotlib.pyplot`
- `seaborn`

### Machine Learning
- `tensorflow` with `keras`
- `EarlyStopping` callback

### Evaluation
- `sklearn.metrics`

### Techniques
- CNN (Convolutional Neural Network)
- ReLU and Softmax activations
- Adam optimizer
- Sparse Categorical Crossentropy loss

---

## Motivation

This dataset encourages research into gesture recognition and real-world ML applications. It can support projects that aim to bridge communication gaps for the deaf and hard-of-hearing using computer vision.

