Для выполнения необходимо создать папку videos_for_classify и поместить в неё видео формата .mp4
Для проверки необходим файл guitar_type.txt

Скрипт classify_instrument.py анализирует видео из папки videos_for_classify и производит проверку результатов на основе файла guitar_type.txt, если этот файл есть, а иначе скрипт выводит результаты анализов

Под Анализом имеется ввиду определение типа инструмента, используемого в видео

Логи запуска
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:00<00:00, 25598.95it/s]
Список полученных файлов для обработки 📁 ✅ 📥
['videos_for_classify\\01_radiohead-creep_unknown.mp4', 'videos_for_classify\\02_eminem-real-slim-shady_unknown.mp4', 'videos_for_classify\\03_imagine-dragons-radioactive_unknown.mp4', 'videos_for_classify\\04_nirvana-te
en-spirit_unknown.mp4', 'videos_for_classify\\05_gravity-falls-opening_unknown.mp4', 'videos_for_classify\\06_tokyo-ghoul-unravel_unknown.mp4', 'videos_for_classify\\07_pirates-hes-a-pirate_unknown.mp4', 'videos_for_classify\\08_aladdin-whole-new-world_unknown.mp4', 'videos_for_classify\\09_mj-billie-jean_unknown.mp4', 'videos_for_classify\\10_mj-beat-it_unknown.mp4', 'videos_for_classify\\11_evangelion-zankoku-tenshi_unknown.mp4']     

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Результат анализа видео 'videos_for_classify\01_radiohead-creep_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 53.83%

Результат анализа видео 'videos_for_classify\02_eminem-real-slim-shady_unknown.mp4':
🎸 ♫ Определен класс: acoustic
🎯 Уверенность ИИ: 86.53%

Результат анализа видео 'videos_for_classify\03_imagine-dragons-radioactive_unknown.mp4':
🎸 ♫ Определен класс: acoustic
🎯 Уверенность ИИ: 83.69%

Результат анализа видео 'videos_for_classify\04_nirvana-teen-spirit_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 87.46%

Результат анализа видео 'videos_for_classify\05_gravity-falls-opening_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 78.47%

Результат анализа видео 'videos_for_classify\06_tokyo-ghoul-unravel_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 81.05%

Результат анализа видео 'videos_for_classify\07_pirates-hes-a-pirate_unknown.mp4':
🎸 ♫ Определен класс: acoustic
🎯 Уверенность ИИ: 74.30%

Результат анализа видео 'videos_for_classify\08_aladdin-whole-new-world_unknown.mp4':
🎸 ♫ Определен класс: acoustic
🎯 Уверенность ИИ: 75.17%

Результат анализа видео 'videos_for_classify\09_mj-billie-jean_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 81.14%

Результат анализа видео 'videos_for_classify\10_mj-beat-it_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 44.78%

Результат анализа видео 'videos_for_classify\11_evangelion-zankoku-tenshi_unknown.mp4':
🎸 ♫ Определен класс: electric
🎯 Уверенность ИИ: 79.93%

Был сформирован список результатов 📁 📝 ✅
01_radiohead-creep_unknown.mp4   electric
02_eminem-real-slim-shady_unknown.mp4    acoustic
03_imagine-dragons-radioactive_unknown.mp4       acoustic
04_nirvana-teen-spirit_unknown.mp4       electric
05_gravity-falls-opening_unknown.mp4     electric
06_tokyo-ghoul-unravel_unknown.mp4       electric
07_pirates-hes-a-pirate_unknown.mp4      acoustic
08_aladdin-whole-new-world_unknown.mp4   acoustic
09_mj-billie-jean_unknown.mp4    electric
10_mj-beat-it_unknown.mp4        electric
11_evangelion-zankoku-tenshi_unknown.mp4         electric

Полученная информация для проверки ✅ 📊 🔍 из файла guitar_type.txt
{'01_radiohead-creep': 'electric', '02_eminem-real-slim-shady': 'acoustic', '03_imagine-dragons-radioactive': 'acoustic', '04_nirvana-teen-spirit': 'electric', '05_gravity-falls-opening': 'electric', '06_tokyo-ghoul-unravel': 'electric', '07_pirates-hes-a-pirate': 'acoustic', '08_aladdin-whole-new-world': 'acoustic', '09_mj-billie-jean': 'electric', '10_mj-beat-it': 'electric', '11_evangelion-zankoku-tenshi': 'electric'}
✅ Проверка пройдена: 01_radiohead-creep -> electric
✅ Проверка пройдена: 02_eminem-real-slim-shady -> acoustic
✅ Проверка пройдена: 03_imagine-dragons-radioactive -> acoustic
✅ Проверка пройдена: 04_nirvana-teen-spirit -> electric
✅ Проверка пройдена: 05_gravity-falls-opening -> electric
✅ Проверка пройдена: 06_tokyo-ghoul-unravel -> electric
✅ Проверка пройдена: 07_pirates-hes-a-pirate -> acoustic
✅ Проверка пройдена: 08_aladdin-whole-new-world -> acoustic
✅ Проверка пройдена: 09_mj-billie-jean -> electric
✅ Проверка пройдена: 10_mj-beat-it -> electric
✅ Проверка пройдена: 11_evangelion-zankoku-tenshi -> electric

Итого успешных проверок: 11
Ошибок: 0
Пропущено (нет в эталоне): 0