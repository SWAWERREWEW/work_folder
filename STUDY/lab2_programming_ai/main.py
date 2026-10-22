# main.py
import pandas

data = pandas.read_csv("data.csv")

print(data.head(20))

"""
(.venv) D:\python_programming\clean_folder\lab2_programming_ai>python main.py
                                               Review  Rating
0   3D Touch просто восхитительная вещь! Заряд дер...       5
1   Отключается при температуре близкой к нулю, не...       4
2   В Apple окончательно решили не заморачиваться,...       3
3   Постарался наиболее ёмко и коротко описать все...       4
4   Достойный телефон. Пользоваться одно удовольст...       5
5                                        6s gold 64gb       5
6   Мой первый айфон. Скажу честно- Эппл ранее ник...       5
7   Мне очень понравилась эта модель! Во-первых кл...       5
8   Долгое время пользовалась iPhone 5s 16Gb. Он м...       5
9   Раньше был Samsung Galaxy Alpha SM-G850F 32Gb,...       4
10  Бесит, что защитные стекла все идут не на весь...       5
11  Вот уже третий год радует телефон, а с ios10 с...       5
12  Телефонами apple пользуюсь с выхода модели 3G ...       4
13  Аккумулятора в стандартном режиме всегда хвата...       5
14  После 5s чувствуется разница в первую очередь ...       5
15  Мне показалось, что телефон очень прочный, рон...       5
16  Покупали сразу два - черный и золотой. Пользую...       5
17  В целом аппарат кайфовый, но цена нынешняя пуг...       5
18  Проблемы чисто софтверные, и выражаются в явно...       3
19  Полный офис яблочников с многолетним стажем. З...       3

(.venv) D:\python_programming\clean_folder\lab2_programming_ai>
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Загрузка данных из CSV
df = pd.read_csv("data.csv")

# ОЧИСТКА: Удаляем пропуски (NaN)
df = df.dropna(subset=["Review", "Rating"])

# ФИЛЬТРАЦИЯ: Оставляем только строки, где рейтинг строго от 1 до 5
# Это уберет ошибочные нули или отрицательные значения из вашего CSV
df = df[df["Rating"].isin([1, 2, 3, 4, 5])]

# Извлекаем тексты и метки
texts = df["Review"].astype(str).values
ratings = df["Rating"].values - 1  # Теперь оценки 1..5 гарантированно станут 0..4

NUM_SAMPLES = len(df)
NUM_CLASSES = 5   # Классы: 0, 1, 2, 3, 4


# 2. Разделение на Train и Test
indices = np.arange(NUM_SAMPLES)
np.random.seed(42)
np.random.shuffle(indices)

texts = texts[indices]
ratings = ratings[indices]

SPLIT_RATIO = 0.8
split_idx = int(NUM_SAMPLES * SPLIT_RATIO)

X_train_raw, X_test_raw = texts[:split_idx], texts[split_idx:]
y_train, y_test = ratings[:split_idx], ratings[split_idx:]

# 3. Векторизация текста штатными средствами TensorFlow
MAX_WORDS = 5000   # Уменьшили словарь (ускорит процесс)
MAX_LEN = 100      # Уменьшили длину отзыва (сильно ускорит LSTM/CNN)

vectorizer = layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

print("--- Начало построения словаря (vectorizer.adapt) ---")
# Превращаем в tf.data.Dataset и обрабатываем батчами по 64 строки — это гораздо быстрее!
train_dataset_raw = tf.data.Dataset.from_tensor_slices(X_train_raw).batch(64)
vectorizer.adapt(train_dataset_raw)
print("--- Словарь успешно построен! ---")

# Превращаем сырой текст в матрицы индексов
X_train = vectorizer(X_train_raw)
X_test = vectorizer(X_test_raw)

# 4. Архитектура модели (адаптированная под 5 классов)
def build_cnn_model():
    """Одномерная сверточная нейросеть (1D CNN)"""
    model = models.Sequential([
        # Слой эмбеддингов
        layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # ВАЖНО: 5 выходов (по числу классов) и функция активации softmax
        layers.Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model

# 5. Компиляция и обучение
model = build_cnn_model()

# ВАЖНО: для мультиклассовой классификации, где метки — целые числа (0-4),
# используется функция потерь sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("--- Обучение модели ---")
model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.1
)

# 6. Оценка качества на Test выборке
print("\n--- Оценка качества на Test ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss (Ошибка): {loss:.4f}")
print(f"Test Accuracy (Точность): {accuracy:.4f}")

# Пример предсказания для нового отзыва
print("\n--- Пример проверки модели ---")
custom_reviews = ["Ужасный телефон, постоянно выключается на морозе!", "Отличный аппарат, пользуюсь с удовольствием."]
custom_reviews_vectorized = vectorizer(np.array(custom_reviews))
predictions = model.predict(custom_reviews_vectorized)

for rev, pred in zip(custom_reviews, predictions):
    # argmax находит индекс максимальной вероятности, прибавляем 1, чтобы вернуть шкалу 1-5
    predicted_rating = np.argmax(pred) + 1 
    print(f"Отзыв: {rev} -> Прогноз рейтинга: {predicted_rating}")
