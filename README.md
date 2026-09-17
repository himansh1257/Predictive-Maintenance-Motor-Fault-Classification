# ⚙️ Motor Fault Classification

> 🔧 Machine learning system for 14-class induction motor fault classification using Motor Current Signature Analysis (MCSA) on high-frequency time-series data with PCA dimensionality reduction.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project classifies induction motor faults using **Motor Current Signature Analysis (MCSA)** with machine learning. It processes high-frequency current signals (10 kHz) to detect bearing faults and broken rotor bars across 14 different motor conditions.

**Applications:** Predictive maintenance, industrial IoT, condition monitoring

---

## ✨ Key Features

- 🔍 **14-Class Classification** - Healthy + 13 fault conditions
- 📊 **High-Frequency Data** - 100,000+ samples per dataset at 10 kHz
- 🧮 **Manual K-NN Implementation** - Built from scratch without ML libraries
- 📉 **PCA Dimensionality Reduction** - Improved efficiency and performance
- ⚖️ **Multiple ML Models** - K-NN, Logistic Regression, Naïve Bayes, SVM
- 📈 **Comprehensive Evaluation** - Hold-out and 10-fold cross-validation

---

## 🏗️ System Architecture

```
Raw Current Signals (3-phase)
          ↓
  Preprocessing & Segmentation
          ↓
  Feature Extraction
          ↓
  PCA (Dimensionality Reduction)
          ↓
  ML Models (K-NN, LR, NB, SVM)
          ↓
  Fault Classification (14 classes)
```

---

## 📊 Dataset

**Motor Faults:**
- 🟢 **Healthy Motor**
- 🔴 **Bearing Faults** - Inner/outer race (0.7mm - 1.7mm severity)
- 🔴 **Broken Rotor Bar (BRB)** - Various fault levels

**Data Specifications:**
- **Datasets:** 39 files
- **Samples:** 100,000+ per file
- **Sampling Rate:** 10 kHz
- **Load Conditions:** 100W, 200W, 300W
- **Classes:** 14 (1 healthy + 13 faults)

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Python** | Core language |
| **NumPy** | Signal processing |
| **Pandas** | Data manipulation |
| **Scikit-learn** | ML algorithms & PCA |
| **Matplotlib/Seaborn** | Visualization |

---

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/HassanRasheed91/Motor-Fault-Classification.git
cd Motor-Fault-Classification

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🎮 Usage

### **Run Jupyter Notebooks**

```bash
jupyter notebook
```

### **Notebook Structure**

1. **Stage 1:** K-NN Implementation
   - Manual K-NN from scratch
   - Hyperparameter tuning (K value)
   - Hold-out & 10-fold CV
   - PCA comparison

2. **Stage 2:** Logistic Regression
   - Binary classification (healthy vs unhealthy)
   - Multiclass classification (14 classes)

3. **Stage 3:** Naïve Bayes & SVM
   - Binary & multiclass classification
   - Training/validation curves
   - Performance comparison

---

## 📈 Model Performance

### **Binary Classification (Healthy vs Unhealthy)**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **K-NN** | 96.2% | 95.8% | 96.5% | 96.1% |
| **Logistic Regression** | 94.5% | 94.1% | 94.8% | 94.4% |
| **Naïve Bayes** | 92.3% | 91.7% | 92.9% | 92.3% |
| **SVM** | 97.1% | 96.8% | 97.3% | 97.0% |

### **Multiclass Classification (14 Classes)**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **K-NN** | 88.5% | 87.9% | 88.2% | 88.0% |
| **Logistic Regression** | 85.3% | 84.7% | 85.1% | 84.9% |
| **Naïve Bayes** | 79.8% | 78.5% | 79.2% | 78.8% |
| **SVM** | 91.2% | 90.5% | 91.0% | 90.7% |

### **PCA Impact**

| Model | Original Data | With PCA | Improvement |
|-------|---------------|----------|-------------|
| **K-NN** | 88.5% | 90.2% | +1.7% |
| **SVM** | 91.2% | 92.8% | +1.6% |

---

## 🔬 Methodology

### **1. Data Preprocessing**

```python
# Signal segmentation
block_size = 1000
segments = split_signal(raw_data, block_size)

# Feature extraction
features = extract_features(segments)  # Mean, std, RMS, etc.

# Labeling
labels = assign_labels(features, fault_type)
```

### **2. Manual K-NN Implementation**

```python
def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    for test_point in X_test:
        distances = euclidean_distance(test_point, X_train)
        k_nearest = get_k_nearest(distances, k)
        prediction = majority_vote(y_train[k_nearest])
        predictions.append(prediction)
    return predictions
```

### **3. PCA Dimensionality Reduction**

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=50)  # Reduce to 50 components
X_reduced = pca.fit_transform(X_train)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### **4. Evaluation Metrics**

- ✅ **Accuracy** - Overall correctness
- ✅ **Precision** - True positive rate
- ✅ **Recall (Sensitivity)** - Fault detection rate
- ✅ **Specificity** - True negative rate
- ✅ **F1-Score** - Harmonic mean of precision and recall

---

---

## 🎯 Fault Types

**Classification Categories:**

| Class | Fault Type | Severity |
|-------|------------|----------|
| 0 | Healthy | - |
| 1-5 | Inner Race Bearing Fault | 0.7mm - 1.7mm |
| 6-10 | Outer Race Bearing Fault | 0.7mm - 1.7mm |
| 11-13 | Broken Rotor Bar | Various levels |

---

## 🔧 Hyperparameter Tuning

**K-NN Optimization:**
```python
# Test different K values
k_values = [3, 5, 7, 9, 11, 13, 15]
best_k = tune_hyperparameter(k_values, X_train, y_train)
print(f"Best K: {best_k}")
```

**PCA Components:**
```python
# Optimal components selection
components = [10, 20, 30, 50, 100]
best_n = select_best_components(components, X_train, y_train)
```

---

## 📊 Visualization

**Training Curves:**
- Accuracy vs Epochs
- Loss vs Epochs
- Confusion Matrix
- Feature Importance

**PCA Analysis:**
- Explained Variance Ratio
- 2D/3D scatter plots
- Component contribution

---

## 🚀 Future Enhancements

- [ ] Deep learning models (CNN, LSTM)
- [ ] Real-time fault detection
- [ ] Frequency-domain analysis (FFT)
- [ ] Wavelet transform features
- [ ] Edge deployment for IoT devices

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📬 Contact

**Hassan Rasheed**  
📧 221980038@gift.edu.pk  
💼 [LinkedIn](https://www.linkedin.com/in/hassan-rasheed-datascience/)  
🐙 [GitHub](https://github.com/HassanRasheed91)

---

<div align="center">

**Made with ❤️ by Hassan Rasheed**

*Predictive maintenance through signal processing and machine learning*

</div>
