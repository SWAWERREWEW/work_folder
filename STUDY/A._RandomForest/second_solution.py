#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


def main():
    # ===== 1) Загрузка датасета (ИМЯ ФАЙЛА ИЗ ЗАДАНИЯ!) =====
    df = pd.read_csv("churnclean.csv")

    # Переименовываем столбцы на случай, если в файле на платформе они другие
    df = df.rename(columns={
        'International plan': 'interplan',
        'IntlPlan': 'interplan',
        'Intl Plan': 'interplan',
        'Voice mail plan': 'vmailplan',
        'VMailPlan': 'vmailplan',
        'VMail Plan': 'vmailplan',
        'Churn': 'churn'
    })

    # ===== 2) Предобработка =====
    # Кодируем булевы признаки
    if df['interplan'].dtype == 'object':
        df['interplan'] = df['interplan'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    else:
        df['interplan'] = df['interplan'].astype(int)

    if df['vmailplan'].dtype == 'object':
        df['vmailplan'] = df['vmailplan'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    else:
        df['vmailplan'] = df['vmailplan'].astype(int)

    if df['churn'].dtype == 'object':
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    else:
        df['churn'] = df['churn'].astype(int)

    # Удаление выбросов (метод перцентилей 1% и 99%)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'churn']
    lower_bound = df[numeric_cols].quantile(0.01)
    upper_bound = df[numeric_cols].quantile(0.99)
    df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

    # Удаляем текстовый State, если он есть
    if 'State' in df.columns:
        df = df.drop(columns=['State'])

    # Разделяем на X и y
    X = df.drop('churn', axis=1)
    y = df['churn']

    # ===== 3) Train/test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    random_state_some = 10

    # ===== 4) RandomForestClassifier + CV =====
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=random_state_some)

    # ВОЗВРАЩАЕМ min_samples_leaf=4 ДЛЯ СТАБИЛЬНОСТИ!
    rf = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=random_state_some,
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
            random_state=random_state_some,
            min_samples_leaf=4  # Тоже добавляем сюда
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
    df = pd.read_csv("churnclean.csv")
    print(df.head(20))

"""churnclean.csv
    Unnamed: 0 state  acclength  areacode interplan vmailplan  vmailmess  daycalls  daycharge  evecalls  evecharge  nightcalls  nightcharge  intlmins  intlcalls  intlcharge  custservcalls churn
0            0    KS        128       415        no       yes         25       110      45.07        99      16.78          91        11.01      10.0          3        2.70              1    no
1            1    OH        107       415        no       yes         26       123      27.47       103      16.62         103        11.45      13.7          3        3.70              1    no
2            2    NJ        137       415        no        no          0       114      41.38       110      10.30         104         7.32      12.2          5        3.29              0    no
3            3    MO        147       415       yes        no          0        79      26.69        94       8.76          96         9.53       7.1          6        1.92              0    no
4            4    IN         65       415        no        no          0       137      21.95        83      19.42         111         9.40      12.7          6        3.43              4   yes
5            5    MT         95       510        no        no          0        88      26.62        75      21.05         115         8.65      12.3          5        3.32              3    no
6            6    VT         93       510        no        no          0       114      32.42       111      18.55         121         5.83       8.1          3        2.19              3    no
7            7    VA         76       510        no       yes         33        66      32.25        65      18.09         108         7.46      10.0          5        2.70              1    no
8            8    TX         73       415        no        no          0        90      38.15        88      13.56          74         8.68      13.0          2        3.51              1    no
9            9    FL        147       415        no        no          0       117      26.37        93      20.37         133         9.40      10.6          4        2.86              0    no
10          10    AZ        130       415        no        no          0       112      31.11        99       6.20          78         8.18       9.5         19        2.57              0    no
11          11    SC        111       415        no        no          0       103      18.77       102      11.67         105         8.53       7.7          6        2.08              2    no
12          12    VA        132       510        no        no          0        86      13.79        72      20.84         115        10.67      10.3          2        2.78              0    no
13          13    MT         54       408        no        no          0        73      22.83       100      13.22          68         4.59      14.7          4        3.97              3    no
14          14    MO         20       415        no        no          0       109      32.30        84      21.95         102         8.17       6.3          6        1.70              0    no
15          15    HI         49       510        no        no          0       117      20.28       109      18.28          90         8.04      11.1          1        3.00              1    no
16          16    IL        142       415        no        no          0        95      14.42        63      11.62         148        11.27      14.2          6        3.83              2    no
17          17    NH         75       510        no        no          0       105      38.44       107      17.13          98        11.08      10.3          5        2.78              1    no
18          18    GA         72       415        no       yes         37        80      37.40       102      18.47          71         6.88      14.7          6        3.97              3    no
19          19    AK         36       408        no       yes         30       128      24.87        80      13.81         109         5.82      14.5          6        3.92              0    no
"""
