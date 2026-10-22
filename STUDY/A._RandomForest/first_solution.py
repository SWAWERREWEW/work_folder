#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

"""
Шаблон решения (Random Forest, churnclean.csv)

ВАЖНО ДЛЯ ПРОВЕРКИ (checker.py):
Нужно напечатать 6 строк в таком порядке:

1) cv_score        (float)
2) cv_stdev        (float)
3) accuracy        (float)
4) rocauc_list     (5 чисел ROC-AUC, в одной строке через пробел)  <-- именно тут, сразу после 3 метрик
5) n_est           (строка: 20 40 60 80 100)
6) optimal_n_est   (одно число: 20/40/60/80/100)

Остальная часть решения — TODO.
"""

import numpy as np
import pandas as pd


def main():
    # ===== 1) Загрузка датасета =====
    df = pd.read_csv("churnclean.csv")

    # Переименовываем столбцы для удобства (как требуется в задании)
    df = df.rename(columns={
        'International plan': 'interplan',
        'Voice mail plan': 'vmailplan',
        'Churn': 'churn'
    })

    # ===== 2) TODO: предобработка =====
    # TODO: удалить выбросы
    # TODO: закодировать interplan, vmailplan, churn булевыми (0/1)
    # TODO: разделить на X и y

    # 2.1) Закодируем булевы признаки (Yes/No и True/False -> 1/0)
    df['interplan'] = df['interplan'].map({'Yes': 1, 'No': 0})
    df['vmailplan'] = df['vmailplan'].map({'Yes': 1, 'No': 0})
    # В датасете BigML Churn может быть True/False или Yes/No, обрабатываем оба варианта
    if df['churn'].dtype == 'object':
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    else:
        df['churn'] = df['churn'].astype(int)

    # 2.2) Удаление выбросов (метод IQR - межквартильный размах)
    # Оставляем только числовые признаки для анализа выбросов
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'churn']

    lower_bound = df[numeric_cols].quantile(0.01)
    upper_bound = df[numeric_cols].quantile(0.99)

    # Оставляем только строки, где все числовые признаки попадают в границы 1% - 99%
    df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

    # 2.3) Удалим текстовый столбец 'State', так как он содержит много уникальных значений
    # и может переобучить модель. (Либо можно было сделать One-Hot Encoding)
    if 'State' in df.columns:
        df = df.drop(columns=['State'])

    # 2.4) Разделяем на X и y
    X = df.drop('churn', axis=1)
    y = df['churn']

    # ===== 3) TODO: train/test split =====
    # TODO: разделить на обучающую и тестовую выборки

    random_state_some = 10

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state_some, stratify=y
    )

    # ===== 4) TODO: RandomForestClassifier + CV =====
    # TODO: создать RandomForestClassifier(n_estimators=50, class_weight="balanced")
    # TODO: cross_val_score(cv=5, scoring="recall") -> cv_score, cv_stdev
    cv_score = 0.0   # TODO
    cv_stdev = 0.0   # TODO

    # Создаем стратегию кросс-валидации с перемешиванием
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_some)

    rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=random_state_some)

    # Кросс-валидация (cv=5, scoring='recall')
    cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring="recall")
    cv_score = cv_scores.mean()
    cv_stdev = cv_scores.std()

    # ===== 5) TODO: accuracy на тесте =====
    accuracy = 0.0   # TODO

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ===== 6) TODO: перебор n_estimators и rocauc_list =====
    n_est = [20, 40, 60, 80, 100]
    rocauc_list = [0.0, 0.0, 0.0, 0.0, 0.0]  # TODO: 5 значений ROC-AUC

    n_est = [20, 40, 60, 80, 100]
    rocauc_list = []
    recall_list = []

    for n in n_est:
        rf_n = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=random_state_some)
        rf_n.fit(X_train, y_train)

        # Вычисляем ROC-AUC
        y_pred_proba = rf_n.predict_proba(X_test)[:, 1]  # Вероятности для класса 1
        rocauc = roc_auc_score(y_test, y_pred_proba)
        rocauc_list.append(rocauc)

        # Вычисляем Recall (нужен для поиска optimal_n_est)
        y_pred_n = rf_n.predict(X_test)
        recall = recall_score(y_test, y_pred_n)
        recall_list.append(recall)

    # ===== 7) TODO: optimal_n_est по максимальному Recall =====
    optimal_n_est = 20  # TODO

    optimal_n_est = n_est[np.argmax(recall_list)]

    # ===== ВЫВОД (не менять порядок!) =====
    print(f"{cv_score:.6f}")
    print(f"{cv_stdev:.6f}")
    print(f"{accuracy:.6f}")
    print(" ".join(f"{v:.6f}" for v in rocauc_list))
    print(" ".join(str(x) for x in n_est))
    print(int(optimal_n_est))


if __name__ == "__main__":
    main()
