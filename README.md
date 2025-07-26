# Machine Learning Project: Advanced Logistic Regression & Model Evaluation with Airbnb Data 🏡📈

## 📚 Overview  
This project builds on logistic regression techniques using the Airbnb NYC listings dataset. You will train and tune logistic regression models, evaluate them using precision-recall and ROC curves, perform feature selection, and save your best model for deployment or future use.

---

## 🎯 Objectives  
By the end of this project, you will:

- Build and prepare your dataset for modeling  
- Create labeled examples and split data into training and testing sets  
- Train and evaluate a logistic regression model with default hyperparameters  
- Use GridSearchCV to find the optimal hyperparameters for logistic regression  
- Train, test, and evaluate the best logistic regression model  
- Plot precision-recall and ROC curves and compute AUC scores for model comparison  
- Practice feature selection using SelectKBest  
- Save your best-performing model as a `.pkl` file for reuse  

---

## 🧠 Problem Statement  
Predict a binary outcome from Airbnb listings, improving model performance through hyperparameter tuning and feature selection. Example:  
> **Can we better predict whether a listing is available year-round?**

---

## 🛠️ Project Steps  

### 1. Data Preparation  
- Load and clean the dataset  
- Define labels and features  
- Split data into train and test sets  

### 2. Model Training & Evaluation  
- Train logistic regression with default hyperparameters  
- Evaluate using accuracy, precision-recall curve, ROC curve, and AUC  

### 3. Hyperparameter Tuning  
- Use `GridSearchCV` to find the best `C` value for logistic regression  
- Retrain and evaluate the tuned model  

### 4. Feature Selection  
- Apply `SelectKBest` to identify most predictive features  
- Retrain model using selected features  

### 5. Model Saving  
- Save the best model as a `.pkl` file for later use  
- Add model and relevant dataset files to your GitHub repo  

---

## 💻 Sample Code Snippet  
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv('data/airbnbData_Prepared.csv')

# Create label
df['is_available_year_round'] = (df['availability_365'] > 300).astype(int)

# Features and labels
X = df.drop(columns=['is_available_year_round', 'availability_365', 'price'])
y = df['is_available_year_round']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# GridSearch for hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train_selected, y_train)

best_model = grid.best_estimator_

# Save best model
joblib.dump(best_model, 'models/best_logistic_model.pkl')

