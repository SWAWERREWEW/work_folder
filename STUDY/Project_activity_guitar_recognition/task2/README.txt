Для работы необходима программа FFmpeg
Скачивание
https://www.gyan.dev/ffmpeg/builds/
Название необходимого архива
ffmpeg-git-essentials.7z
Нужно сохранить или запомнить полный путь до папки с файлом ffmpeg.exe

После разархивирования необходимо добавить в систему
win → Измененение системых переменных среды → Переменные среды... → Переменные среды пользователя → Path → Изменить или нажать два раза на Path → Создать → Сохранённый ранее путь до файла ffmpeg.exe (путь нужно указать без файла ffmpeg.exe) → ОК

Необходима версия Python 3.11 (установка должна происходить от имени администратор)
winget install Python.Python.3.11

Проверка
python --version
или при наличии Python Launcher
py -3.11 --version

Можно создать виртуальное окружение
python -m venv .venv
.\.venv\Scripts\activate.bat

Установка необходимых зависимостей для запуска
pip install uv
uv sync

Запуск
python app.py
или
fastapi dev app.py

Открывается по ссылке в браузере
http://127.0.0.1:8000/docs


Для выполнения скрипта classify_instrument.py необходимо создать папку videos_for_classify и поместить в неё видео формата .mp4
Для проверки необходим файл guitar_type.txt

https://drive.google.com/drive/folders/102YC-IPFdMnfzcZAJYflD7F91dTkt501

Скрипт classify_instrument.py анализирует видео из папки videos_for_classify и производит проверку результатов на основе файла guitar_type.txt, если этот файл есть, а иначе скрипт выводит результаты анализов

Под Анализом имеется ввиду определение типа инструмента, используемого в видео

Логи запуска
(.venv) D:\python_programming\for_git_repositories\guitar-fingering-ai\classify_instrument>python classify_instrument.py
Список полученных файлов для обработки:
['videos_for_classify\\video_1.mp4', 'videos_for_classify\\video_10.mp4', 'videos_for_classify\\video_11.mp4', 'videos_for_classify\\video_12.mp4', 'videos_for_classify\\video_2.mp4', 'videos_for_classify\\video_3.mp4', 'videos_for_classify\\video_4.mp4', 'videos_for_classify\\video_5.mp4', 'videos_for_classify\\video_6.mp4', 'videos_for_classify\\video_7.mp4', 'videos_for_classify\\video_8.mp4', 'videos_for_classify\\video_9.mp4']
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 590/590 [00:00<00:00, 770.60it/s]
D:\python_programming\for_git_repositories\guitar-fingering-ai\classify_instrument\classify_instrument.py:192: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:821.)
  class_scores[cls] = class_scores.get(cls, 0.0) + float(prompt_probs[prompt_idx])

Результат анализа видео 'videos_for_classify\video_1.mp4':
Определен класс: electric (уверенность: 93%)
[h264 @ 000002ce239dc740] Missing reference picture, default is 0
[h264 @ 000002ce239dc740] decode_slice_header error
...
Результат анализа видео 'videos_for_classify\video_7.mp4':
Определен класс: classical (уверенность: 95%)

Результат анализа видео 'videos_for_classify\video_8.mp4':
Определен класс: electric (уверенность: 90%)

Результат анализа видео 'videos_for_classify\video_9.mp4':
Определен класс: electric (уверенность: 88%)
[OK] video_1: electric
[OK] video_10: acoustic
[OK] video_11: classical
[OK] video_12: electric
[OK] video_2: electric
[ERR] video_3: предсказано 'acoustic', ожидалось 'classical'
[OK] video_4: electric
[OK] video_5: classical
[OK] video_6: acoustic
[ERR] video_7: предсказано 'classical', ожидалось 'electric'
[OK] video_8: electric
[ERR] video_9: предсказано 'electric', ожидалось 'classical'

Итого: успешно=9, ошибок=3, пропущено=0
