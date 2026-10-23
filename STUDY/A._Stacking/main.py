import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. Загрузка данных
df = pd.read_csv('forest_covertype.csv')

# 2. Очистка данных
# Удаление дубликатов
df = df.drop_duplicates()

# Удаление выбросов (пример: по IQR для числовых признаков)
# Здесь показан общий подход: для каждого числового столбца (кроме Id и целевого)
# можно отсечь значения за пределами 1.5 * IQR.
# Для ускорения и простоты можно применить только к основным признакам,
# но в реальной задаче лучше проверить каждый признак.
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Cover_Type'])
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 3. Проверка зависимости признаков и удаление высококоррелированных
# Вычисляем матрицу корреляций для числовых признаков
corr_matrix = df[numeric_cols].corr().abs()
# Выбираем верхний треугольник матрицы
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Находим признаки с корреляцией выше порога (например, 0.95)
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(columns=to_drop)
# Обновляем список числовых признаков после удаления
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Cover_Type'])

# 4. Кодирование категориальных признаков
# В данном датасете все признаки, кроме целевого, обычно числовые.
# Если есть object-колонки, их нужно закодировать (например, One-Hot Encoding).
# Проверим наличие object-колонок:
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 5. Формирование матрицы признаков и целевой переменной
# Id не является признаком
X = df.drop(columns=['Id', 'Cover_Type'])
y = df['Cover_Type']

# Кодируем целевую переменную (Cover_Type) в числа от 0 до n_classes-1
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Разделение на обучающую и тестовую выборки
# Используем стратификацию, так как классы могут быть несбалансированы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Создание стэка моделей RandomForestClassifier
# Базовые модели: несколько RandomForestClassifier с разными гиперпараметрами
base_models = [
    ('rf1', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
    ('rf2', RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', random_state=42, n_jobs=-1)),
    ('rf3', RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1))
]

# Мета-модель: логистическая регрессия (хорошо работает как мета-классификатор для вероятностей)
# meta_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
meta_model = LogisticRegression(max_iter=1000)

# Создание StackingClassifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,          # внутренняя кросс-валидация для получения мета-признаков
    n_jobs=-1,
    passthrough=False
)

# 8. Обучение классификатора
stacking_model.fit(X_train, y_train)

# 9. Оценка качества по ROC-AUC
# Получаем вероятности классов на тестовой выборке
y_proba = stacking_model.predict_proba(X_test)

# Вычисляем ROC-AUC для многоклассовой задачи с усреднением OvR
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
print(f"ROC-AUC (OvR, macro): {roc_auc:.4f}")

# Вывод AUC с усреднением 'weighted' для сравнения
roc_auc_weighted = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
print(f"ROC-AUC (OvR, weighted): {roc_auc_weighted:.4f}")
