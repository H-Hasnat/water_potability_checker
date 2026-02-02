# 💧 Water Potability Prediction Project

This project aims to predict whether water is safe for human consumption based on its chemical properties. I have explored multiple machine learning models and optimized them to ensure reliable safety predictions.

---

## 🏗️ 1. Initial Exploration (Model Selection)
At the beginning of the project, I trained and evaluated a diverse set of classification models to establish a baseline:

* **Linear Models:** Logistic Regression
* **Tree-Based Models:** Random Forest, Decision Tree, Gradient Boosting
* **Kernel Models:** SVM (Support Vector Machine)
* **Distance Models:** KNN (K-Neighbors Classifier)
* **Ensemble Methods:** Stacking and Voting Classifiers
* **Advanced Boosting:** XGBoost



---

## 🛠️ 2. Optimization & Refining
After the primary testing, I selected **XGBoost** and **Random Forest** for further development. To improve the results, I performed:
1.  **Cross-Validation:** To verify model stability across different data splits.
2.  **Hyperparameter Tuning:** Systematically adjusting parameters to find the best configuration.

---

## ⚖️ 3. Solving Class Imbalance (SMOTE)
During evaluation, I noticed that the **Recall** scores for Class 0 (Unsafe) and Class 1 (Safe) were inconsistent. To fix this:
* Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.
* Implemented **ImbPipeline** (Imbalanced-learn Pipeline) to ensure proper preprocessing without data leakage.



---

## 🎯 4. Final Performance & Threshold Tuning
The final step was to maximize the model's reliability. I focused on **Probability Threshold Tuning** for the XGBoost model. 
* **Best Threshold:** `0.52`
* **Conclusion:** After tuning, **XGBoost** emerged as the best-performing model, providing a balanced F1-score and higher accuracy.



---

## 📊 Evaluation Summary
| Model | Optimization | Result |
| :--- | :--- | :--- |
| **XGBoost** | SMOTE + Threshold Tuning | **Best Performance** |
| **Random Forest** | Hyperparameter Tuning | Stable Performance |
