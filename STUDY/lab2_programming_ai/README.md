# Перед выполнением необходимо загрузить data.csv


Нужно выбрать любой набор данных на русском языке для задачи классификации и решить на нем задачу классификации текста.
<br>
(Примеры phonereviews, sentiment-analysis-dataset)
<br>
Ограничения:
- Можно использовать только полносвязные нейросети, одномерные сверточные, LSTM или GRU.
- Нельзя использовать предобученные сети, в том числе различные варианты BERT или LLM.
- На train и test делите сами, качество оцениваете на test.
- Подход к оценке качества нужно будет объяснить на защите.
- Нужно использовать Tensorflow (Keras) или Pytorch, сторонние фреймворки для обучения не допускаются


# Вывод
```powershell
(.venv) D:\python_programming\clean_folder\lab2_programming_ai>python main.py
D:\python_programming\clean_folder\lab2_programming_ai\main.py:9: SyntaxWarning: invalid escape sequence '\p'
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
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792419318.743826   13052 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792419320.578338   13052 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792419324.034181   13052 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
--- Начало построения словаря (vectorizer.adapt) ---
--- Словарь успешно построен! ---
D:\python_programming\clean_folder\lab2_programming_ai\.venv\Lib\site-packages\keras\src\layers\core\embedding.py:123: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
--- Обучение модели ---
Epoch 1/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m64s←[0m 6ms/step - accuracy: 0.5701 - loss: 1.0332 - val_accuracy: 0.5935 - val_loss: 0.9504
Epoch 2/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m64s←[0m 6ms/step - accuracy: 0.6180 - loss: 0.9065 - val_accuracy: 0.6264 - val_loss: 0.8936
Epoch 3/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m62s←[0m 6ms/step - accuracy: 0.6735 - loss: 0.7904 - val_accuracy: 0.6494 - val_loss: 0.8613
Epoch 4/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m62s←[0m 6ms/step - accuracy: 0.7255 - loss: 0.6778 - val_accuracy: 0.6808 - val_loss: 0.8732
Epoch 5/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m62s←[0m 6ms/step - accuracy: 0.7674 - loss: 0.5858 - val_accuracy: 0.6984 - val_loss: 0.8541
Epoch 6/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m61s←[0m 6ms/step - accuracy: 0.8002 - loss: 0.5091 - val_accuracy: 0.7205 - val_loss: 0.9078
Epoch 7/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m61s←[0m 6ms/step - accuracy: 0.8251 - loss: 0.4502 - val_accuracy: 0.7298 - val_loss: 0.8967
Epoch 8/8
←[1m10302/10302←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m61s←[0m 6ms/step - accuracy: 0.8447 - loss: 0.4029 - val_accuracy: 0.7408 - val_loss: 0.9427

--- Оценка качества на Test ---
Test Loss (Ошибка): 0.9457
Test Accuracy (Точность): 0.7373

--- Пример проверки модели ---
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 62ms/step
Отзыв: Ужасный телефон, постоянно выключается на морозе! -> Прогноз рейтинга: 1
Отзыв: Отличный аппарат, пользуюсь с удовольствием. -> Прогноз рейтинга: 5

(.venv) D:\python_programming\clean_folder\lab2_programming_ai>
```


# Для защиты
```
Этот подход (оценка на отложенной выборке Test Accuracy / Loss) выбран потому, что он является золотым стандартом в машинном обучении и идеально подходит под ваши жесткие условия задачи: обучить модель без использования сторонних ML-фреймворков (вроде scikit-learn или PyTorch).
Вот ключевые причины, почему выбран именно такой вариант:

1. Защита от «зубрёжки» (переобучения)

Если оценивать качество нейросети на тех же данных, на которых она училась (Train), метрики будут идеальными. Но это не значит, что сеть поняла человеческий язык — она могла просто запомнить конкретные слова из обучающих отзывов.
• Оценка на Test выборке, которую модель вообще не видела в процессе обучения, показывает её реальную обобщающую способность — то, как она будет работать на практике со свежими отзывами из интернета.

2. Реализация чистыми средствами Python и TensorFlow

Поскольку вы просили обойтись без сторонних библиотек, мы разделили данные стандартным срезом массивов NumPy (texts[:split_idx]), а саму оценку доверили встроенному методу model.evaluate(). Это позволило:
• Избежать импорта sklearn.model_selection.train_test_split.
• Сделать код максимально легковесным, быстрым и независимым от окружения.

3. Комплексный взгляд: Ошибка + Точность

Метод model.evaluate() возвращает сразу две важнейшие метрики:
• Loss (Функция потерь / Ошибка): Показывает, насколько сильно модель «уверена» в своих ошибках. Если Loss падает от эпохи к эпохе, значит, нейросеть учится распределять вероятности правильно (например, она начинает понимать разницу между оценками 4 и 5).
• Accuracy (Точность): Самая понятная метрика для мультиклассовой классификации. Она показывает простое отношение количества правильных ответов к общему числу отзывов в тестовом наборе. Например, Accuracy: 0.65 означает, что модель угадала точную оценку (от 1 до 5) в 65% случаев.

4. Внутренняя валидация (validation_split=0.1)

В коде во время обучения используется параметр validation_split. Он заставляет Keras прямо во время обучения отрезать еще 10% от тренировочных данных, чтобы показывать вам метрики в конце каждой эпохи. Это нужно, чтобы вы вовремя заметили, если модель пойдет «не туда» (например, если точность на обучении растет, а на валидации падает — пора останавливать процесс).
Поскольку у вас большой датасет, метрика Accuracy очень показательна. Скажите:
• До какого значения точности (accuracy) дошла ваша модель на тесте?
• Хотите ли вы посмотреть, как заменить архитектуру CNN на LSTM или GRU в этом же коде, чтобы сравнить их метрики качества на вашем датасете?
```
