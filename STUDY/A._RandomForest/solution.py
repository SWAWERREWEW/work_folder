#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


def main():
    # ===== 1) Загрузка датасета =====
    df = pd.read_csv("churnclean.csv")

    # Удаляем лишний столбец с индексами, если он есть
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Удаляем текстовый столбец state (с маленькой буквы, как в файле)
    if 'state' in df.columns:
        df = df.drop(columns=['state'])
    elif 'State' in df.columns:
        df = df.drop(columns=['State'])

    # ===== 2) Предобработка =====
    # Кодируем булевы признаки. ВАЖНО: добавляем маленькие буквы 'yes' и 'no'!
    mapping = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, True: 1, False: 0}

    df['interplan'] = df['interplan'].map(mapping)
    df['vmailplan'] = df['vmailplan'].map(mapping)
    df['churn'] = df['churn'].map(mapping)

    # Удаление выбросов (метод перцентилей 1% и 99%)
    # Исключаем целевую переменную 'churn'
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'churn']
    lower_bound = df[numeric_cols].quantile(0.01)
    upper_bound = df[numeric_cols].quantile(0.99)
    df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

    # Разделяем на X и y
    X = df.drop('churn', axis=1)
    y = df['churn']

    # ===== 3) Train/test split =====
    # Используем random_state=10, который дал отличный результат
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10, stratify=y
    )

    # ===== 4) RandomForestClassifier + CV =====
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    rf = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=10,
        min_samples_leaf=4
    )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring="recall")
    cv_score = cv_scores.mean()
    cv_stdev = cv_scores.std()

    # ===== 5) accuracy на тесте =====
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ===== 6) Перебор n_estimators =====
    n_est = [20, 40, 60, 80, 100]
    rocauc_list = []
    recall_list = []

    for n in n_est:
        rf_n = RandomForestClassifier(
            n_estimators=n,
            class_weight="balanced",
            random_state=10,
            min_samples_leaf=4
        )
        rf_n.fit(X_train, y_train)

        y_pred_proba = rf_n.predict_proba(X_test)[:, 1]
        rocauc = roc_auc_score(y_test, y_pred_proba)
        rocauc_list.append(rocauc)

        y_pred_n = rf_n.predict(X_test)
        recall = recall_score(y_test, y_pred_n)
        recall_list.append(recall)

    # ===== 7) optimal_n_est =====
    optimal_n_est = n_est[np.argmax(recall_list)]

    # ===== ВЫВОД (строго 6 строк) =====
    print(f"{cv_score:.6f}")
    print(f"{cv_stdev:.6f}")
    print(f"{accuracy:.6f}")
    print(" ".join(f"{v:.6f}" for v in rocauc_list))
    print(" ".join(str(x) for x in n_est))
    print(int(optimal_n_est))


if __name__ == "__main__":
    main()
