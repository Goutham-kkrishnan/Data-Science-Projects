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

|Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|--------------|----------|-----|-----------|--------|----------|------|
|Logistic_Regression|0.8597122302158273|0.9842740614199484|0.8341264121699034|0.8376137932811766|0.8350115033059174|0.8343814652613227|
|Decision_Tree|0.9820143884892086|0.9874546107480305|0.9797464440321584|0.9778496013790131|0.978487194250741|0.9787938044980558|
|KNN|0.8165467625899281|0.9521763613175971|0.7853127298664824|0.7902967269197482|0.7836502097776636|0.784191983849469|
|Naive_Bayes|0.762589928057554|0.97586425377299|0.8258094440122089|0.7200486276800523|0.710093212508377|0.730935267875705|
|Random_Forest|0.9964028776978417|0.9999635787683661|0.9964285714285713|0.9952380952380953|0.9957703742299323|0.9957592204635709|
|XGBoost|0.9928057553956835|0.9999291387628558|0.9910364145658264|0.9910364145658264|0.9910364145658264|0.9914891011511143|


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
