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
    # target_col = ...

    # === TODO 2: разделить данные на X (признаки) и y_raw (целевая переменная) ===
    X = df.drop('species', axis=1)
    y_raw = df["species"]
    # X = ...
    # y_raw = ...

    # === TODO 3: закодировать целевую переменную через LabelEncoder ===
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # y = ...

    # === TODO 4: создать модели (можно/нужно использовать Pipeline + StandardScaler) ===
    lr_model = StandardScaler().fit_transform(df)
    svc_model = SVC(random_state=0, gamma='auto')
    knn_model = KNeighborsClassifier(n_neighbors=5).fit(X, y_raw)
    # lr_model = ...
    # svc_model = ...
    # knn_model = ...

    # === TODO 5: оценить каждую модель через cross_val_score (cv=10, scoring="accuracy") ===
    lr = cross_val_score.fit(lr_model, X, y, cv=10, scoring="accuracy")
    svc = cross_val_score.fit(svc_model, X, y, cv=10, scoring="accuracy")
    knn = cross_val_score.fit(knn_model, X, y, cv=10, scoring="accuracy")

    # === TODO 6: собрать список estimators для VotingClassifier ===
    estimators = [('lr', lr), ('svc', svc),('knn', knn)]
    # estimators = ...

    # === TODO 7: VotingClassifier voting="hard" + кросс-валидация ===
    hard_voting = VotingClassifier(estimators=estimators, voting='hard')
    hard_v_cls = cross_val_score.fit(hard_voting, X, y, cv=10, scoring="accuracy")
    # hard_voting = ...
    # hard_v_cls = ...

    # === TODO 8: VotingClassifier voting="soft" + кросс-валидация ===
    soft_voting = VotingClassifier(estimators=estimators, voting='soft')
    soft_v_cls = cross_val_score.fit(soft_voting, X, y, cv=10, scoring="accuracy")

    # === TODO 9: сформировать сетку весов для GridSearchCV (взвешенное голосование) ===
    weight_grid = GridSearchCV(df, estimators, cv=10)

    # === TODO 10: GridSearchCV для VotingClassifier(voting="soft") и получение лучшей точности ===
    grid = GridSearchCV(hard_voting, estimators)
    # grid.fit(...)
    accuracy_gridcv = grid.fit(weight_grid)

    # === ВАЖНО: строку вывода НЕ менять (6 чисел в строгом порядке) ===
    print(f"{lr:.6f} {svc:.6f} {knn:.6f} {hard_v_cls:.6f} {soft_v_cls:.6f} {accuracy_gridcv:.6f}")


if __name__ == "__main__":
    main()
