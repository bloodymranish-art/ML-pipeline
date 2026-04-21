# Automated End-to-End ML Pipeline

## Overview
This repository contains a statistically grounded, production-ready Machine Learning pipeline designed for structured data. The project implements a rigorous mathematical framework for **Empirical Risk Minimization (ERM)** and **Hypothesis Space Reduction**, achieving high accuracy (~95%) with controlled bias-variance trade-offs.

The pipeline is modeled as a functional transformation $X' = S(T(X))$, where data undergoes automated imputation ($T$) and distribution-aware scaling ($S$) before entering a pruned feature space.

## Key Features
- **Statistical Transformation Pipeline:** Automated handling of numerical and categorical features using `ColumnTransformer`.
- **Hypothesis Space Reduction:**
  - **Variance Thresholding:** $\text{Var}(X_j) > \epsilon$ to remove low-information noise.
  - **Correlation Pruning:** A custom transformer (`CorrelationDropper`) to eliminate multicollinearity and improve model identifiability.
  - **Mutual Information Maximization:** Selection of features based on target dependency using `SelectKBest`.
- **ERM Optimization:** Comprehensive hyperparameter search across multiple dimensions (Learning Rate, Depth, Feature Count) using Stratified $k$-fold Cross-Validation.
- **Model Interpretability:** Integration with **SHAP** (SHapley Additive exPlanations) for local and global feature importance analysis.
- **Robust Evaluation:** Implementation of ROC-AUC, Precision-Recall curves, and detailed classification profiling.

## Mathematical Framework
The goal is to find a function $f$ in the hypothesis space $\mathcal{H}$ that minimizes the empirical risk:
$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i))$$
where $L$ is the loss function (Log-Loss for XGBoost).

## Project Structure
- `Automated_ML_Pipeline.ipynb`: The core notebook containing the step-by-step implementation.
- `automated_ml_pipeline.py`: The Python script version for production/automation.
- `automated_ml_pipeline.pkl`: Serialized model for downstream inference and deployment.

## Setup & Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost shap joblib matplotlib
   python automated_ml_pipeline.py
   Performance MetricsAccuracy: ~95%Identifiability: High (via correlation and variance pruning).Generalization: Verified via $k$-fold cross-validation to ensure stability on unseen distributions.Tech StackLanguages: PythonLibraries: Scikit-learn, XGBoost, SHAP, NumPy, Pandas, JoblibVisualization: Matplotlib, Seaborn
