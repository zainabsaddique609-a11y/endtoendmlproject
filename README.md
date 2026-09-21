# End-to-End Machine Learning Project - Task 3

## 📌 Project Overview
This project implements an end-to-end machine learning pipeline using the California Housing dataset (containing over 20,000 rows). The primary objective is to build, optimize, and evaluate multiple regression models to predict median house values, complete with data preprocessing, feature engineering, hyperparameter tuning, model performance comparison, and production serialization.

---

## 🛠️ Technologies & Libraries Used
* **Python**
* **Pandas & NumPy** (Data Manipulation & Preprocessing)
* **Scikit-Learn** (Machine Learning Models, Scaling & Hyperparameter Tuning)
* **Matplotlib & Seaborn** (Data Visualization)
* **Joblib** (Model Serialization)

---

## 🚀 Project Workflow & Steps
1. **Data Collection & Preparation:** Loaded the dataset, handled missing values, removed duplicates, and performed feature engineering (creating `RoomsPerHousehold`). Scaled features using `StandardScaler`.
2. **Model Training:** Trained four different regression algorithms:
   * Linear Regression
   * Decision Tree Regressor
   * Random Forest Regressor
   * Gradient Boosting Regressor
3. **Model Optimization:** Applied `GridSearchCV` for hyperparameter tuning and evaluated models using 5-fold cross-validation.
4. **Model Serialization:** Saved the best-performing optimized model as `best_model.pkl` using `joblib`.

---

## 📊 Visualizations & Results
* **Model Comparison:** Evaluated models based on R2 Score to identify the best performer.
* **Feature Importance:** Analyzed which attributes contributed most heavily to the Random Forest model's predictions.
* **Actual vs Predicted:** Visualized the correlation between actual housing values and model predictions.

---

## 📁 Repository Files
* `best_model.pkl`: The serialized best-performing machine learning model.
* `model_comparison_task3.png`: Bar chart comparing the performance of all trained models.
* `feature_importance.png`: Feature importance analysis plot.
* `actual_vs_predicted.png`: Scatter plot showing actual versus predicted target values.
*
