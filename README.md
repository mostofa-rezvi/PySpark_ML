# PySpark Machine Learning Assignment

**Student Name:** Mostofa Aminur Rashid  
**Student ID:** 2506107  
**Submission Date:** July 6, 2026  

---

## Overview

This repository contains the implementation of two machine learning assignments using **Apache PySpark (MLlib)**:

- **Part A:** Supervised Learning — Titanic Survival Prediction
- **Part B:** Unsupervised Learning — Mall Customer Segmentation

The final report is available in [`PySpark_ML_Assignment_Report.pdf`](./PySpark_ML_Assignment_Report.pdf).

---

## Repository Structure

```
PySpark_ML/
├── part_a_titanic.py          # Part A: Random Forest classifier on Titanic dataset
├── part_b_mall.py             # Part B: K-Means clustering on Mall Customers dataset
├── generate_report.py         # Script to generate the PDF assignment report
├── download_data.py           # Script to download required datasets
├── data/
│   ├── titanic.csv            # Titanic passenger dataset
│   └── Mall_Customers.csv     # Mall customer dataset
├── outputs/
│   ├── titanic_confusion_matrix.png
│   ├── titanic_feature_importances.png
│   ├── titanic_results.txt
│   ├── mall_clusters_k3.png
│   ├── mall_clusters_k5.png
│   └── mall_results.txt
└── PySpark_ML_Assignment_Report.pdf   # Final compiled report
```

---

## Part A: Titanic Survival Prediction (Supervised Learning)

- **Algorithm:** Random Forest Classifier (100 trees, max depth = 5)
- **Train/Test Split:** 80% / 20%
- **Key Results:**

| Metric | Value |
|---|---|
| Accuracy | 82.76% |
| Weighted Precision | 83.39% |
| Weighted Recall | 82.76% |
| Weighted F1-Score | 82.35% |
| AUC (ROC) | 89.12% |

---

## Part B: Mall Customer Segmentation (Unsupervised Learning)

- **Algorithm:** K-Means Clustering (k = 3 and k = 5)
- **Features:** Age, Annual Income (k$), Spending Score (1-100)
- **Preprocessing:** StandardScaler for feature normalization

---

## How to Run

### Prerequisites
- Python 3.8+
- Java JDK 17 (required for PySpark)
- PySpark, scikit-learn, matplotlib, seaborn, fpdf2

### Steps

```bash
# 1. Install dependencies
pip install pyspark scikit-learn matplotlib seaborn fpdf2

# 2. Download datasets
python download_data.py

# 3. Run Part A (Titanic)
python part_a_titanic.py

# 4. Run Part B (Mall Clustering)
python part_b_mall.py

# 5. Generate the PDF report
python generate_report.py
```

---

## Platform
- Apache Spark 4.1.2 (via PySpark)
- Java JDK 17 LTS
