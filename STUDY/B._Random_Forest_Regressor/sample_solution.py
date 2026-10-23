#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# TODO: добавьте нужные импорты из sklearn:
# - train_test_split
# - ColumnTransformer / OneHotEncoder (или другой способ кодирования)
# - RandomForestRegressor
# - (опционально) Pipeline


def main():
    # === 1) Загрузка данных ===
    df = pd.read_csv("Salaries.csv")

    # === 2) Предобработка данных ===
    # TODO: удалить дубликаты
    # df = ...

    # TODO: удалить выбросы (выберите метод, например IQR или z-score)
    # df = ...

    # TODO: разделить на признаки и целевую переменную salary_in_usd
    # X = ...
    # y = ...

    # TODO: разделить на обучающую и тестовую выборки
    # X_train, X_test, y_train, y_test = ...

    # TODO: закодировать признаки с типом object
    # X_train = ...
    # X_test  = ...
    # (или сделать это через Pipeline + ColumnTransformer)

    # === 3) Модель ===
    # TODO: создать RandomForestRegressor с n_estimators=10 и oob_score=True
    # model = ...

    # TODO: обучить модель
    # model.fit(...)

    # === 4) OOB score ===
    # TODO: получить oob_score и сохранить в переменную oob_score
    # oob_score = ...

    # ВАЖНО: чекер берёт первое float-число из вывода.
    # Поэтому выводим только oob_score одним числом.
    print(oob_score)


if __name__ == "__main__":
    main()
