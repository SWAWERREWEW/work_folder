import pandas as pd, os

df = pd.read_csv("submission.csv", header=None)
print("shape:", df.shape)                  # должно быть (2424, 2)
print(df.head())
print("классы:", set(df[1]))
print("пропуски:", df.isna().sum().sum())
print("строк с пробелами:", (df[0] != df[0].str.strip()).sum())

test_files = set(
    f for f in os.listdir("./dataset/test")
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)
sub_files = set(df[0])
print("нет в submission:", len(test_files - sub_files))
print("лишние в submission:", len(sub_files - test_files))

import os
test_files = set(os.listdir("./dataset/test"))
for f in ["n04308397_5230.png", "n04583620_8407.png",
          "n02686121_2618.png", "n03596543_5074.png"]:
    print(f, "->", f in test_files)
