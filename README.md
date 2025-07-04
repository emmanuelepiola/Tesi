# Sensor-Based Chemical Classification System 🧪

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)](https://scikit-learn.org)
[![Thesis](https://img.shields.io/badge/Project-Thesis-orange.svg)]()

## 📖 Overview

This repository contains a complete machine learning pipeline for chemical classification using sensor data. The system employs advanced Random Forest models with sophisticated data balancing techniques and comprehensive visualization tools, including **confusion matrix analysis**.

## 🎯 Key Features

### 🧠 **Advanced Machine Learning Pipeline**
- **Random Forest Classifier** with hyperparameter optimization
- **Ensemble Methods** (XGBoost, LightGBM, CatBoost, ExtraTrees)
- **Stacking Ensemble** with meta-learner for enhanced accuracy

### ⚖️ **Comprehensive Data Balancing**
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **BorderlineSMOTE** for borderline cases
- **Hybrid methods** (SMOTE + Tomek Links, SMOTE + ENN)
- **Target-specific balancing** for minority classes

### 🔧 **Advanced Feature Engineering**
- **Ratio features** between sensor types
- **Polynomial features** for non-linear relationships
- **Statistical features** (mean, std, interactions)
- **Manual feature selection** (16 optimized features)

### 📊 **Comprehensive Evaluation & Visualization**
- **Confusion Matrix Visualizations**:
  - Sample counts matrix
  - Recall-focused (normalized by true class)
  - Precision-focused (normalized by predicted class)
  - Comprehensive 2x2 subplot view
- **Feature importance analysis**
- **Cross-validation** with stratified K-fold
- **Class-wise performance metrics**
- **Error pattern analysis**

### 🎯 **Chemical Detection**
Supports classification of **25 chemical compounds**:
- **Acids**: Acetic Acid, Formic Acid, Phosphoric Acid
- **Alcohols**: Ethanol, Isopropanol, Bioethanol
- **Gases**: Ammonia, Methane, Butane, Water Vapor
- **Industrial**: Diesel, Gasoline, Kerosene, Lighter Fluid
- **And more...**

## 📁 **Project Structure**

```
Tesi/
├── model.py                 # Core ML model implementation
├── evaluation.py            # Interactive evaluation system
├── EDA.ipynb               # Exploratory Data Analysis
├── .gitignore              # Git ignore rules
├── visualizations/         # Generated plots and confusion matrices
│   ├── confusion_matrix_comprehensive.png
│   ├── confusion_matrix_counts.png
│   ├── confusion_matrix_recall.png
│   ├── confusion_matrix_precision.png
│   └── feature_importance.png
└── README.md               # This file
```

## 🚀 **Quick Start**

### **1. Interactive Evaluation**
```bash
python3 evaluation.py
```

This launches an interactive menu where you can:
- Configure feature engineering options
- Choose balancing methods
- Set up cross-validation
- Compare multiple configurations
- View results history

### **2. Direct Model Usage**
```python
from model import SensorRandomForestModel

# Initialize model
rf_model = SensorRandomForestModel()

# Load and prepare data
data = rf_model.load_data()
X_train, X_test, y_train, y_test = rf_model.prepare_data(
    data, 
    feature_engineering=True,
    balance_classes=True,
    balance_method='smote'
)

# Train model
rf_model.train_model(X_train, y_train)

# Evaluate with confusion matrix
accuracy, predictions = rf_model.evaluate_model(X_test, y_test)
```

## 📊 **Confusion Matrix Features**

The system automatically generates **4 types of confusion matrices**:

1. **📊 Counts Matrix**: Raw prediction counts
2. **🎯 Recall Matrix**: Performance per true class (sensitivity)
3. **📈 Precision Matrix**: Performance per predicted class (specificity)  
4. **🔍 Comprehensive View**: All matrices in one 2x2 subplot

### **Advanced Analysis**
- **Problematic class identification** (precision/recall < 80%)
- **Most confused pairs** analysis
- **Color-coded visualizations** for easy interpretation
- **High-resolution exports** (300 DPI) for presentations

## ⚖️ **Balancing Strategies**

| Method | Description | Best For |
|--------|------------|----------|
| **SMOTE** | Synthetic oversampling | General imbalance |
| **ADASYN** | Adaptive synthetic sampling | Varying class densities |
| **BorderlineSMOTE** | Focus on borderline cases | Difficult boundaries |
| **Target Classes** | Balance specific minorities | Known problematic classes |
| **Aggressive** | Multi-stage balancing | Severe imbalance |

## 🔬 **Research Applications**

This system is designed for:
- **Chemical detection** and identification
- **Sensor data analysis** and pattern recognition
- **Imbalanced dataset** handling in classification
- **Multi-class classification** optimization
- **Feature engineering** for sensor data

## 🛠️ **Technical Requirements**

- **Python 3.8+**
- **Core Libraries**: scikit-learn, pandas, numpy
- **ML Libraries**: xgboost, lightgbm, catboost
- **Visualization**: matplotlib, seaborn
- **Imbalanced Learning**: imbalanced-learn

## 📈 **Performance Metrics**

The system provides comprehensive evaluation:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Cross-validation** scores with confidence intervals
- **Confusion matrix** analysis with error patterns
- **Feature importance** rankings
- **Training/testing** time measurements

## 🎓 **Academic Context**

This project is part of a thesis on **machine learning applications in chemical sensing**, focusing on:
- Advanced classification techniques for sensor data
- Handling class imbalance in chemical detection
- Visualization and interpretability of ML models
- Comparative analysis of ensemble methods

## 📧 **Contact**

**Emmanuel Epiola**  
📍 Thesis Project - Chemical Classification System  
🔗 [GitHub Repository](https://github.com/emmanuelepiola/Tesi)

---

*🔬 Advancing chemical detection through machine learning and data science* 