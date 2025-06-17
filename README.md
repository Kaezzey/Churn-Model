# Customer Churn Prediction Using a Deep Feedforward Neural Network (F1 = 0.74)

A robust machine learning model for predicting customer churn using a deep feedforward neural network (FNN).  
Achieves a **0.74 F1-score** on the churn class with **0.85 accuracy**, averaged over **100 randomized splits** (Monte Carlo Cross-Validation).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Keras](https://img.shields.io/badge/Keras-API-red)
![Status](https://img.shields.io/badge/F1%20Score-0.74-brightgreen)

---

## Evaluation

**Monte Carlo Cross-Validation** (100 randomized train/test splits)  
Average performance on test sets:

<img width="408" alt="image" src="https://github.com/user-attachments/assets/eb7f8285-1fcf-4212-8617-9005b22e04d5" />


---

## Model Architecture

**Type:** Deep Feedforward Neural Network  
**Framework:** Keras + TensorFlow backend  

```python
Input Layer:         47 features
Hidden Layer 1:     128 units + ReLU + BatchNorm + Dropout(0.6)
Hidden Layer 2:      64 units + ReLU + BatchNorm + Dropout(0.6)
Output Layer:         1 unit (sigmoid activation)
Loss:               Binary Crossentropy
Optimizer:          Adam
