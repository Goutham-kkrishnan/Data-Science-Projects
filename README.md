## a. Problem Statement

The aim is to develop and evaluate multiple machine learning classification models to predict obesity levels based on individuals’ eating habits, physical activity, and lifestyle conditions.

The dataset **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** contains demographic, behavioral, and physical attributes such as age, gender, dietary habits, physical activity frequency, and transportation methods.  
The target variable **NObeyesdad** represents different obesity categories.

The primary goal is to compare the performance of various machine learning models using standard evaluation metrics and identify the most accurate and reliable model for predicting obesity levels.

---

## b. Dataset Description

The dataset **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** is obtained from the **UCI Machine Learning Repository**:  
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

### Dataset Characteristics
- Total instances: **2111**
- Total attributes: **17**
- Data collected from individuals in **Mexico, Peru, and Colombia**
- Age range: approximately **14–61 years**
- Combination of **survey-based data** and **synthetically generated samples** (using SMOTE)

### Feature Categories
- **Demographic Features**: Gender, Age, Height, Weight  
- **Eating Habits**: High-calorie food consumption, vegetable intake frequency, number of meals, snacking habits, water consumption  
- **Lifestyle Factors**: Physical activity frequency, technology usage time, alcohol consumption, smoking habits, transportation method  
- **Family History**: Family history with overweight  

### Target Variable
- **NObeyesdad**: A multi-class label representing seven obesity levels:
  - Insufficient Weight  
  - Normal Weight  
  - Overweight Level I  
  - Overweight Level II  
  - Obesity Type I  
  - Obesity Type II  
  - Obesity Type III  

---

## c. Models Used

The following six machine learning classification models were implemented and evaluated:

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

These models cover linear, probabilistic, distance-based, tree-based, and ensemble learning approaches.

### Evaluation Metrics Comparison Table

|Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|--------------|----------|-----|-----------|--------|----------|------|
|Logistic_Regression|0.8597|0.9842|0.8341|0.8376|0.8350|0.8343|
|Decision_Tree|0.9820|0.9874|0.9797|0.9778|0.9784|0.9787|
|KNN|0.8165|0.9521|0.7853|0.7902|0.7836|0.7841|
|Naive_Bayes|0.7625|0.9758|0.8258|0.7200|0.7100|0.7309|
|Random_Forest|0.9964|0.9999|0.9964|0.9952|0.9957|0.9957|
|XGBoost|0.9928|0.9999|0.9910|0.9910|0.9910|0.9914|

### Model-wise Performance Observations

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | Provided strong baseline performance with high AUC but limited ability to model non-linear feature interactions. |
| Decision Tree | Achieved very high accuracy and balanced metrics, indicating strong learning capacity but potential overfitting risk. |
| kNN | Showed moderate performance and was sensitive to feature scaling and choice of k value. |
| Naive Bayes | Simple and computationally efficient with good AUC, but lower recall and F1 score due to independence assumptions. |
| Random Forest | Best overall performer with superior accuracy, precision, recall, F1 score, and MCC, indicating excellent generalization. |
| XGBoost | Delivered near-optimal and consistent performance across all evaluation metrics, demonstrating robustness and efficiency. |
