# 🩺 Pneumonia Detection using Balanced ResNet and Explainable AI

This project focuses on detecting **Pneumonia from chest X-ray images** using a **Balanced ResNet model**, and integrates **Explainable AI (XAI)** techniques to interpret model predictions.  
The aim is to enhance **trust and transparency** in AI-driven medical diagnostics.

---

## 🧠 Project Overview

This project uses a **deep learning–based image classification model** to automatically detect pneumonia from chest X-rays.  
A **ResNet** architecture was trained on a **balanced dataset** to ensure fair performance across both classes (*Normal* and *Pneumonia*).  

To make the model interpretable, **Explainable AI (XAI)** methods such as **Grad-CAM** and **SHAP** are applied to visualize which parts of the X-ray influenced the model’s decision.

---

## 📁 Directory Structure

├── data/ # Contains training, validation, and test datasets

├── app.py # Streamlit/Flask app for running the model interactively

├── evaluate_model.py # Script to evaluate model accuracy, recall, precision, F1-score

├── optimal_threshold.txt # Stores the optimal threshold determined from validation metrics

├── pneumonia_model_balanced.h5 # Trained ResNet model (balanced dataset)

├── requirements.txt # List of Python dependencies

├── train_balanced_resnet.py # Script for training ResNet on balanced dataset

├── training_results.png # Visualization of training/validation accuracy and loss

├── xai_utils.py # Utility functions for Explainable AI (Grad-CAM, SHAP, etc.)

## ⚙️ Installation and Setup
1. Install dependencies
pip install -r requirements.txt

2. Download the dataset

You can use the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
👉 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## 🧩 Technologies Used

- **Programming Language:** Python 3.x  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Libraries:** NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn  
- **Explainability:** SHAP, Grad-CAM  
- **Web Framework:** Streamlit or Flask  

---

## 🧬 Key Features

- Balanced dataset training to prevent bias  
- Explainable AI (XAI) integration for medical transparency  
- Real-time inference with web app  
- Model evaluation and threshold optimization  
- Visualization of training metrics  

---

## 🧑‍💻 Author

**Shreyash Yenkar**  
🎓 *B.E. in Artificial Intelligence & Data Science*  
📧 [shreyash.y14@gmail.com](mailto:shreyash.y14@gmail.com)  

🔗 **LinkedIn:** [www.linkedin.com/in/shreyash-yenkar](https://www.linkedin.com/in/shreyash-yenkar)

---


## ❤️ Acknowledgments

Special thanks to:

- **Kaggle** for providing the Chest X-Ray Pneumonia dataset  
- **TensorFlow/Keras team** for their powerful deep learning tools  
- **The open-source community** for explainability libraries like SHAP and Grad-CAM  
