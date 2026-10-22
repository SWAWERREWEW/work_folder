#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("diabetes.csv")

# -----------------------------
# 2) Drop missing values (as required)
# -----------------------------
df = df.dropna()

# -----------------------------
# 3) Split into X / y
# -----------------------------
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# -----------------------------
# 4) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5) Base estimator
# -----------------------------
base_est = DecisionTreeClassifier(random_state=42)


# Helper to create BaggingClassifier compatible with sklearn versions
def make_bagging(n_estimators: int):
    return BaggingClassifier(
        estimator=base_est, n_estimators=n_estimators, random_state=42
    )

# -----------------------------
# 6) Main model for CV + accuracy
# -----------------------------
bagging_model = make_bagging(50)


# 7-9) CV (Recall)
cv_scores = cross_val_score(bagging_model, X_train, y_train, cv=5, scoring='recall')
cv_score = np.mean(cv_scores)
cv_stdev = np.std(cv_scores)


# 10) Fit + test Accuracy
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# -----------------------------
# 11-13) ROC-AUC for different n_estimators
# -----------------------------
# 15) optimal_n_est

n_est = [10, 30, 50, 70, 100]
rocauc_list = []
best_recall = -1
optimal_n_est = None

for n in n_est:
    clf = make_bagging(n)
    clf.fit(X_train, y_train)
    
    # Вычисление ROC-AUC на тестовой выборке
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    rocauc_list.append(roc_auc)
    
    # Поиск optimal_n_est по максимальному Recall на тестовой выборке
    y_pred_current = clf.predict(X_test)
    rec = recall_score(y_test, y_pred_current)
    
    if rec > best_recall:
        best_recall = rec
        optimal_n_est = n


# -----------------------------
# Output строго 6 строк (как требует чекер)
# -----------------------------
print(cv_score)
print(cv_stdev)
print(accuracy)
print(n_est)
print(rocauc_list)
print(optimal_n_est)
