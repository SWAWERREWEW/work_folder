#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# TODO: добавьте нужные импорты из sklearn:
# - train_test_split
# - ColumnTransformer / OneHotEncoder (или другой способ кодирования)
# - RandomForestRegressor
# - (опционально) Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def main():
    # === 1) Загрузка данных ===
    df = pd.read_csv("Salaries.csv")
    df = df.drop_duplicates()

    if "salary_in_usd" in df.columns:
        target = "salary_in_usd"
    elif "TotalPayBenefits" in df.columns:
        target = "TotalPayBenefits"
    elif "TotalPay" in df.columns:
        target = "TotalPay"
    else:
        raise ValueError("Не найдена целевая колонка")

    # === 2) Предобработка данных ===
    # TODO: удалить дубликаты
    # df = ...

    df = df.dropna(subset=[target]).copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    # TODO: удалить выбросы (выберите метод, например IQR или z-score)
    # df = ...

    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    df = df[
        (df[target] >= q1 - 1.5 * iqr) &
        (df[target] <= q3 + 1.5 * iqr)
        ]

    # TODO: разделить на признаки и целевую переменную salary_in_usd
    # X = ...
    # y = ...

    X = df.drop(columns=[target])
    y = df[target]

    # TODO: разделить на обучающую и тестовую выборки
    # X_train, X_test, y_train, y_test = ...

    service = {"Id", "EmployeeName", "Notes", "Agency", "Status"}

    if target == "TotalPayBenefits":
        leak = {"TotalPay", "OvertimePay", "OtherPay"}
    elif target == "TotalPay":
        leak = {"TotalPayBenefits", "BasePay", "OvertimePay", "OtherPay"}
    else:
        leak = {
            "TotalPay", "TotalPayBenefits",
            "BasePay", "OvertimePay", "OtherPay", "Benefits",
        }

    forbidden = service | leak
    drop_from_X = [c for c in forbidden if c in X.columns]
    X = X.drop(columns=drop_from_X)

    assert target not in X.columns, "цель попала в признаки"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TODO: закодировать признаки с типом object
    # X_train = ...
    # X_test  = ...
    # (или сделать это через Pipeline + ColumnTransformer)
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # === 3) Модель ===
    # TODO: создать RandomForestRegressor с n_estimators=10 и oob_score=True
    # model = ...
    model = RandomForestRegressor(
        n_estimators=10,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # TODO: обучить модель
    # model.fit(...)
    pipe.fit(X_train, y_train)

    # === 4) OOB score ===
    # TODO: получить oob_score и сохранить в переменную oob_score
    # oob_score = ...
    oob_score = pipe.named_steps["model"].oob_score_

    # ВАЖНО: чекер берёт первое float-число из вывода.
    # Поэтому выводим только oob_score одним числом.
    print(oob_score)


if __name__ == "__main__":
    main()
