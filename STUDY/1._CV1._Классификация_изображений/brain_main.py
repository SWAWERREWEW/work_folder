import tensorflow
import pandas

data = pandas.read_csv("_yandex_submit_cv1.csv")
print(data.head(10))

"""
Получена информация об наборе данных.
   cifar10-train-25538.png  airplane
0   cifar10-train-1086.png  airplane
1  cifar10-train-12192.png  airplane
2  cifar10-train-43783.png  airplane
3       n03596543_2495.png  airplane
4      n03335030_17850.png  airplane
5    cifar10-test-6980.png  airplane
6       n02704645_9275.png  airplane
7       n03596543_1943.png  airplane
8  cifar10-train-27415.png  airplane
9       n04308273_4856.png  airplane
"""

"""
Дальше нужно обработать данные для обучения или создать конвейер данных (Data Pipeline)
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Создаем генератор и говорим ему: "дели все пиксели на 255"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


