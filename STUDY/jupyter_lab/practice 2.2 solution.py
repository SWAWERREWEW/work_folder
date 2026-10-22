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
    target_col = df["species"]

    # === TODO 2: разделить данные на X (признаки) и y_raw (целевая переменная) ===
    X = df.drop(columns=["species"])
    y_raw = df["species"]

    # === TODO 3: закодировать целевую переменную через LabelEncoder ===
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # === TODO 4: создать модели (можно/нужно использовать Pipeline + StandardScaler) ===
    lr_model = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    svc_model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, kernel="rbf"))])
    knn_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3))])

    # === TODO 5: оценить каждую модель через cross_val_score (cv=10, scoring="accuracy") ===
    lr = cross_val_score(lr_model, X, y, cv=10, scoring="accuracy").mean()
    svc = cross_val_score(svc_model, X, y, cv=10, scoring="accuracy").mean()
    knn = cross_val_score(knn_model, X, y, cv=10, scoring="accuracy").mean()

    # === TODO 6: собрать список estimators для VotingClassifier ===
    estimators = [("lr", lr_model), ("svc", svc_model), ("knn", knn_model)]

    # === TODO 7: VotingClassifier voting="hard" + кросс-валидация ===
    hard_voting = VotingClassifier(estimators=estimators, voting="hard")
    hard_v_cls = cross_val_score(hard_voting, X, y, cv=10, scoring="accuracy").mean()

    # === TODO 8: VotingClassifier voting="soft" + кросс-валидация ===
    soft_voting = VotingClassifier(estimators=estimators, voting="soft")
    soft_v_cls = cross_val_score(soft_voting, X, y, cv=10, scoring="accuracy").mean()

    # === TODO 9: сформировать сетку весов для GridSearchCV (взвешенное голосование) ===
    weight_grid = {"weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 1], [1, 2, 2], [2, 1, 2]]}

    # === TODO 10: GridSearchCV для VotingClassifier(voting="soft") и получение лучшей точности ===
    grid = GridSearchCV(estimator=soft_voting, param_grid=weight_grid, cv=10, scoring="accuracy")
    # grid.fit(...)
    accuracy_gridcv = grid.fit(X, y).best_score_

    # === ВАЖНО: строку вывода НЕ менять (6 чисел в строгом порядке) ===
    print(f"{lr:.6f} {svc:.6f} {knn:.6f} {hard_v_cls:.6f} {soft_v_cls:.6f} {accuracy_gridcv:.6f}")


if __name__ == "__main__":
    main()
