# Breast Cancer Diagnosis Using Support Vector Machine (SVM)

## Overview

This project implements a **Support Vector Machine (SVM)** classifier to
predict breast cancer diagnosis as **benign** or **malignant** using
medical diagnostic data. The objective is to apply a powerful supervised
learning algorithm for accurate classification and understand its
performance in medical prediction tasks.

The workflow includes data preprocessing, exploratory data analysis,
model training, and evaluation.

------------------------------------------------------------------------

## Problem Statement

Develop a machine learning model capable of diagnosing breast cancer
based on tumor characteristics.

Objectives: - Analyze medical dataset features - Preprocess and prepare
data for modeling - Train an SVM classification model - Evaluate
classification performance

------------------------------------------------------------------------

## Dataset Description

The dataset contains measurements derived from breast cancer cell nuclei
images.

**Key Features Include:**

  Feature       Description
  ------------- ----------------------------------------
  Radius        Mean distance from center to perimeter
  Texture       Variation in gray-scale values
  Perimeter     Tumor boundary measurement
  Area          Tumor area size
  Smoothness    Radius variation
  Compactness   Shape compactness
  Symmetry      Tumor symmetry
  Diagnosis     Target variable (Benign / Malignant)

------------------------------------------------------------------------

## Project Workflow

### 1. Data Loading

-   Dataset imported using Pandas
-   Dataset inspected using `head()`, `info()`, and statistical
    summaries

### 2. Data Preprocessing

-   Removed unnecessary columns
-   Handled missing values
-   Encoded diagnosis labels
-   Feature scaling applied (important for SVM)
-   Train-test split performed

### 3. Exploratory Data Analysis

-   Feature distribution visualization
-   Correlation heatmap analysis
-   Relationship study between features

### 4. Model Training

-   Support Vector Machine (SVM) classifier implemented
-   Kernel-based classification applied

### 5. Model Evaluation

Performance evaluated using: - Accuracy Score - Confusion Matrix -
Classification Report

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

------------------------------------------------------------------------

## Project Structure

    ├── breast-cancer-diagnosis-svm task-7.ipynb
    ├── breast-cancer.csv
    └── README.md

------------------------------------------------------------------------

## Installation and Usage

### 1. Clone Repository

    git clone <repository-link>

### 2. Install Dependencies

    pip install pandas numpy matplotlib seaborn scikit-learn

### 3. Run Notebook

    jupyter notebook "breast-cancer-diagnosis-svm task-7.ipynb"

Run all cells sequentially.

------------------------------------------------------------------------

## Key Learnings

-   Support Vector Machine classification concepts
-   Importance of feature scaling
-   Kernel-based decision boundaries
-   Model evaluation techniques

------------------------------------------------------------------------

## Future Improvements

-   Hyperparameter tuning using GridSearchCV
-   Comparison with other classifiers
-   Cross-validation implementation
-   Deployment as a medical prediction interface

------------------------------------------------------------------------

## Author

**Anish Sai Nalam**\
B.Tech Computer Science Engineering (AI & ML) Student\
ICFAI University Hyderabad

------------------------------------------------------------------------

## License

This project is created for educational and learning purposes.
