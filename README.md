💧 Water Potability Prediction Project
This project aims to predict the potability of water based on its chemical properties using various Machine Learning classification techniques. The primary focus was on achieving a high-performance model while maintaining a balance between Precision and Recall through advanced optimization.

🚀 Project Workflow
1. Initial Model Exploration
To find the best-performing algorithm, I initially experimented with a wide range of classification models:

Linear Models: Logistic Regression

Tree-Based Models: Decision Tree, Random Forest, Gradient Boosting, XGBoost

Distance/Kernel Models: KNN, Support Vector Machine (SVM)

Ensemble Methods: Stacking and Voting Classifiers

2. Model Selection & Optimization
After evaluating the initial results, XGBoost and Random Forest were selected as the primary candidates due to their superior performance. I then performed:

K-Fold Cross-Validation: To ensure the stability and reliability of the scores across different data splits.

Hyperparameter Tuning: Systematic adjustment of parameters (like max_depth, n_estimators, and learning_rate) to maximize the models' predictive power.

3. Handling Class Imbalance
During evaluation, I noticed a significant discrepancy in the Recall scores between Class 0 (Unsafe) and Class 1 (Safe). To solve this class imbalance, I implemented:

SMOTE (Synthetic Minority Over-sampling Technique): To synthetically generate samples for the minority class.

Imbalanced-learn Pipeline (ImbPipeline): To ensure that over-sampling only occurred during training, preventing data leakage into the validation sets.

4. Threshold Tuning & Final Performance
To further enhance the model's reliability—specifically to minimize the risk of predicting unsafe water as safe—I performed Probability Threshold Tuning.

By adjusting the decision threshold from the default 0.5 to 0.52, I achieved the optimal balance for the XGBoost model, which emerged as the final best-performing model for this project.

📊 Key Results
Final Model: Tuned XGBoost

Best Threshold: 0.52

Key Performance Metrics: Improved F1-Score and balanced Recall across both classes.
