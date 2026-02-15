## a. Problem Statement

The aim of this project is to develop and evaluate multiple machine learning classification models to predict obesity levels based on individuals’ eating habits, physical activity, and lifestyle conditions.

The dataset **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** contains demographic, behavioral, and physical attributes such as age, gender, dietary habits, physical activity frequency, and transportation methods.  
The target variable **NObeyesdad** represents different obesity categories.

The primary goal is to compare the performance of various machine learning models using standard evaluation metrics and identify the most accurate and reliable model for predicting obesity levels.

---

## c. Models Used

The following six machine learning models were implemented and evaluated:

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

These models represent a mix of linear, tree-based, probabilistic, distance-based, and ensemble learning techniques.

---

## Model Performance Comparison

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score |
|--------------|----------|-----|-----------|--------|----------|
| Logistic Regression | 0.8705 | 0.9843 | 0.8492 | 0.8524 | 0.8495 |
| Decision Tree | 0.9928 | 0.9952 | 0.9911 | 0.9916 | 0.9911 |
| kNN | 0.8201 | 0.9528 | 0.7911 | 0.7954 | 0.7878 |
| Naive Bayes | 0.7626 | 0.9771 | 0.8264 | 0.7191 | 0.7088 |
| Random Forest (Ensemble) | 0.9964 | 0.9999 | 0.9964 | 0.9949 | 0.9956 |
| XGBoost (Ensemble) | 0.9928 | 0.9999 | 0.9910 | 0.9910 | 0.9910 |

---

### Model-wise Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Delivered good baseline performance with high AUC but was limited in modeling complex non-linear relationships. |
| Decision Tree | Achieved very high accuracy and balanced metrics, indicating strong learning capability but potential overfitting. |
| kNN | Showed moderate performance and was sensitive to feature scaling and choice of k value. |
| Naive Bayes | Simple and fast model with good AUC, but lower recall and F1 score due to strong independence assumptions. |
| Random Forest (Ensemble) | Best overall performer with highest accuracy, precision, recall, and F1 score, showing excellent generalization. |
| XGBoost (Ensemble) | Delivered near-optimal performance across all metrics and proved to be robust and efficient. |

---

### Conclusion

Ensemble-based models outperformed individual classifiers in predicting obesity levels.  
The Random Forest model achieved the best overall performance, making it the most suitable model for this task, while XGBoost also provided highly competitive results.
