#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    print("⭐")
    print(df.head(10))

    # ===== 2) TODO: предобработка =====
    # TODO: удалить выбросы
    # TODO: закодировать interplan, vmailplan, churn булевыми (0/1)
    # TODO: разделить на X и y

    # ===== 3) TODO: train/test split =====
    # TODO: разделить на обучающую и тестовую выборки

    # ===== 4) TODO: RandomForestClassifier + CV =====
    # TODO: создать RandomForestClassifier(n_estimators=50, class_weight="balanced")
    # TODO: cross_val_score(cv=5, scoring="recall") -> cv_score, cv_stdev
    cv_score = 0.0   # TODO
    cv_stdev = 0.0   # TODO

    # ===== 5) TODO: accuracy на тесте =====
    accuracy = 0.0   # TODO

    # ===== 6) TODO: перебор n_estimators и rocauc_list =====
    n_est = [20, 40, 60, 80, 100]
    rocauc_list = [0.0, 0.0, 0.0, 0.0, 0.0]  # TODO: 5 значений ROC-AUC

    # ===== 7) TODO: optimal_n_est по максимальному Recall =====
    optimal_n_est = 20  # TODO

    # ===== ВЫВОД (не менять порядок!) =====
    print(f"{cv_score:.6f}")
    print(f"{cv_stdev:.6f}")
    print(f"{accuracy:.6f}")
    print(" ".join(f"{v:.6f}" for v in rocauc_list))
    print(" ".join(str(x) for x in n_est))
    print(int(optimal_n_est))


if __name__ == "__main__":
    main()
