import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

# 1. Загружаем предобученную модель CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Промпты: для каждого класса — несколько вариантов (prompt ensembling)
PROMPT_TEMPLATES = {
    "acoustic": [
        "a photo of an acoustic guitar",
        "an image of a person playing an acoustic guitar",
        "a wooden acoustic guitar on a stand",
    ],
    "electric": [
        "a photo of an electric guitar",
        "an electric guitar with an amplifier",
        "a person playing an electric guitar on stage",
    ],
    "bass": [
        "a photo of a bass guitar",
        "a close-up of a bass guitar",
        "a musician playing a bass guitar",
    ],
}

ALL_PROMPTS = []
PROMPT_TO_CLASS = []
for cls, prompts in PROMPT_TEMPLATES.items():
    for p in prompts:
        ALL_PROMPTS.append(p)
        PROMPT_TO_CLASS.append(cls)


def extract_frames(video_path, num_frames=10):
    """Достает num_frames равномерно распределенных кадров из видео"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        raise ValueError(f"❌ Видео слишком короткое: {total_frames} кадров, нужно {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def classify_video(video_path):
    """Определяет тип инструмента по видео с использованием ансамбля промптов"""
    try:
        frames = extract_frames(video_path, num_frames=10)
    except Exception as e:
        print(f"Ошибка чтения видео {video_path}: {e}")
        return None, 0.0

    inputs = processor(text=ALL_PROMPTS, images=frames, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image               # (num_frames, num_prompts)
    probs = logits.softmax(dim=-1).numpy()
    avg_probs_per_prompt = np.mean(probs, axis=0)   # (num_prompts,)

    class_scores = {}
    for prompt_idx, cls in enumerate(PROMPT_TO_CLASS):
        class_scores[cls] = class_scores.get(cls, 0.0) + avg_probs_per_prompt[prompt_idx]

    best_class = max(class_scores, key=class_scores.get)
    total = sum(class_scores.values())
    confidence = class_scores[best_class] / total if total > 0 else 0.0
    return best_class, confidence


def normalize_name(name: str) -> str:
    """
    Приводит имя к единому виду для сравнения:
    - отбрасывает папку (берёт только имя файла);
    - отбрасывает расширение;
    - отбрасывает суффикс _unknown (даже если он с пробелами рядом);
    - убирает пробелы по краям.
    """
    name = str(name).strip()
    # Универсально: работаем и с "\", и с "/"
    name = name.replace("\\", "/").split("/")[-1]
    # Убираем расширение
    if "." in name:
        name = name.rsplit(".", 1)[0]
    name = name.strip()
    # Убираем суффикс _unknown
    if name.endswith("_unknown"):
        name = name[:-len("_unknown")]
    return name.strip()


if __name__ == "__main__":
    folder_path = Path("videos_for_classify")
    files_list = [str(f) for f in folder_path.iterdir() if f.is_file()]

    print("Список полученных файлов для обработки 📁 ✅ 📥")
    print(files_list)
    print()

    results = {}

    for one_video in files_list:
        label, conf = classify_video(one_video)
        if label:
            file_name = Path(one_video).name   # только имя файла, без папки
            results[file_name] = label
            print(f"\nРезультат анализа видео '{one_video}':")
            print(f"🎸 ♫ Определен класс: {label}")
            print(f"🎯 Уверенность ИИ: {conf * 100:.2f}%")

    print("\nБыл сформирован список результатов 📁 📝 ✅")
    for name_song, verdict in results.items():
        print(name_song, "\t", verdict)

    # --- Исправленная проверка ---
    file_for_check = "guitar_type.txt"
    data_for_check = {}

    try:
        with open(file_for_check, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line.strip() or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                # Первое значение строки -> нормализуем так же, как результаты модели
                name = normalize_name(parts[0])
                verdict = parts[-1].strip()
                if name and verdict:
                    data_for_check[name] = verdict
        print(f"\nПолученная информация для проверки ✅ 📊 🔍 из файла {file_for_check}")
        print(data_for_check)
    except FileNotFoundError:
        print(f"\nНе найдена информация для проверки в файле {file_for_check} ❓ 🔍 📊")
        data_for_check = {}

    # Сравнение с нормализацией имён
    if data_for_check:
        success = 0
        errors = 0
        missing = 0

        for name_song, verdict in results.items():
            key = normalize_name(name_song)
            if key in data_for_check:
                expected = data_for_check[key]
                if verdict == expected:
                    success += 1
                    print(f"✅ Проверка пройдена: {key} ♫⚡ {verdict}")
                else:
                    errors += 1
                    print(f"❌ Несовпадение: {key} \t ♫⚡ предсказано '{verdict}', ожидалось '{expected}'")
            else:
                missing += 1
                print(f"⚠️  Нет эталона для: {key}")

        print(f"\nИтого успешных проверок: {success}")
        print(f"Ошибок: {errors}")
        print(f"Пропущено (нет в эталоне): {missing}")
    else:
        print("Проверка не удалась ❌ 🔍 📖 (файл с эталонами не найден или пуст)")
