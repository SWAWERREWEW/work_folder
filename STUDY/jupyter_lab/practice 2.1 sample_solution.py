# main.py (шаблон для студентов)

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


def main():
    # === ВАЖНО: строку загрузки датасета НЕ менять ===
    df = pd.read_csv("iris.csv")

    # === TODO 1: определить целевой столбец (например, 'species' или последний столбец) ===
    target_col = ...

    # === TODO 2: разделить данные на X (признаки) и y_raw (целевая переменная) ===
    X = ...
    y_raw = ...

    # === TODO 3: закодировать целевую переменную через LabelEncoder ===
    le = LabelEncoder()
    y = ...

    # === TODO 4: создать модели (можно/нужно использовать Pipeline + StandardScaler) ===
    lr_model = ...
    svc_model = ...
    knn_model = ...

    # === TODO 5: оценить каждую модель через cross_val_score (cv=10, scoring="accuracy") ===
    lr = ...
    svc = ...
    knn = ...

    # === TODO 6: собрать список estimators для VotingClassifier ===
    estimators = ...

    # === TODO 7: VotingClassifier voting="hard" + кросс-валидация ===
    hard_voting = ...
    hard_v_cls = ...

    # === TODO 8: VotingClassifier voting="soft" + кросс-валидация ===
    soft_voting = ...
    soft_v_cls = ...

    # === TODO 9: сформировать сетку весов для GridSearchCV (взвешенное голосование) ===
    weight_grid = ...

    # === TODO 10: GridSearchCV для VotingClassifier(voting="soft") и получение лучшей точности ===
    grid = ...
    # grid.fit(...)
    accuracy_gridcv = ...

    # === ВАЖНО: строку вывода НЕ менять (6 чисел в строгом порядке) ===
    print(f"{lr:.6f} {svc:.6f} {knn:.6f} {hard_v_cls:.6f} {soft_v_cls:.6f} {accuracy_gridcv:.6f}")


if __name__ == "__main__":
    main()
