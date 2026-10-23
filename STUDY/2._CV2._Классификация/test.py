import pandas as pd

# Читаем текущий файл
df = pd.read_csv("submission.csv")

# 1. Убеждаемся, что нет лишних пробелов
df["FILE_NAME"] = df["FILE_NAME"].str.strip()

# 2. (Опционально) Если нужно, меняем расширение
# df["FILE_NAME"] = df["FILE_NAME"].str.replace(".jpeg", ".jpg")

# 3. Сохраняем с жестким форматом Unix и без BOM
df.to_csv("submission.csv", index=False, encoding="utf-8", lineterminator="\n")

print("Файл пересохранен в строгом формате!")
