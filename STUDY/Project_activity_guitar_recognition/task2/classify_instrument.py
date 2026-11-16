from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Можно попробовать "openai/clip-vit-large-patch14" для большего качества
# MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_NAME = "openai/clip-vit-large-patch14"

# ------------------------------------------------------------
# 1. Расширенные промпты: больше дискриминативных признаков
# ------------------------------------------------------------
PROMPT_TEMPLATES = {
    "acoustic": [
        "a photo of a steel-string acoustic guitar with a large hollow wooden body and a round soundhole",
        "a person strumming a dreadnought acoustic guitar with a pick and a pickguard",
        "a wooden acoustic guitar with six metal strings and a visible soundhole",
        "an acoustic guitar being played with a pick, showing a large body and a pickguard",
        "a close-up of a steel-string acoustic guitar body with a soundhole and a pickguard",
    ],
    "classical": [
        "a photo of a classical Spanish guitar with a round soundhole and no pickguard",
        "a musician playing a nylon-string classical guitar fingerstyle",
        "a traditional wooden classical guitar with a slotted headstock and a wide neck",
        "a classical guitar with nylon strings, a wide fretboard and no pickguard",
        "a close-up of a classical guitar body with a round soundhole and no pickguard",
    ],
    "electric": [
        "a photo of a solid-body electric guitar like a Stratocaster or Les Paul",
        "a guitarist playing a thin solid-body electric guitar with magnetic pickups",
        "a close-up of an electric guitar body with volume knobs and a metal bridge",
        "an electric guitar with a solid body, magnetic pickups and no soundhole",
        "a person playing a electric guitar with a thin body and visible pickups",
        "a solid-body electric guitar with six thin strings and magnetic pickups",
    ],
    "bass": [
    "a photo of a bass guitar with exactly four thick metal strings",
    "a close-up of a bass guitar fretboard with four very thick strings, wider spacing than a guitar",
    "a musician playing a four-string electric bass with a very long scale length",
    "a bass guitar with four thick strings and large tuning pegs, distinctly larger than an electric guitar",
    "a photo of a bass guitar body with a single long pickup and four strings",
    ],
}

ALL_PROMPTS: list[str] = []
PROMPT_TO_CLASS: list[str] = []
for cls, prompts in PROMPT_TEMPLATES.items():
    for prompt in prompts:
        ALL_PROMPTS.append(prompt)
        PROMPT_TO_CLASS.append(cls)


@lru_cache(maxsize=1)
def _get_model_and_processor() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def _extract_frames(video_path: str | Path, num_frames: int = 20) -> list[Image.Image]:
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Видео слишком короткое: {total_frames} кадров, нужно минимум {num_frames}"
        )

    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames: list[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()

    if not frames:
        raise ValueError(f"Не удалось извлечь кадры из видео: {video_path}")
    return frames


def _apply_tta(image: Image.Image, mode: Literal["orig", "flip", "center_crop"]) -> Image.Image:
    """Простые аугментации для TTA."""
    if mode == "orig":
        return image
    if mode == "flip":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if mode == "center_crop":
        w, h = image.size
        crop = int(min(w, h) * 0.85)
        left = (w - crop) // 2
        top = (h - crop) // 2
        return image.crop((left, top, left + crop, top + crop))
    raise ValueError(f"Unknown TTA mode: {mode}")


def _encode_frames(
    frames: list[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    use_tta: bool = True,
) -> torch.Tensor:
    modes = ["orig", "flip"] if use_tta else ["orig"]
    per_frame_embeds: list[torch.Tensor] = []

    for frame in frames:
        aug_embeds = []
        for mode in modes:
            aug_img = _apply_tta(frame, mode)  # type: ignore[arg-type]
            inputs = processor(images=[aug_img], return_tensors="pt")
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                img_emb = model.visual_projection(vision_out.pooler_output)
            aug_embeds.append(F.normalize(img_emb, dim=-1))
        per_frame_embeds.append(torch.stack(aug_embeds, 0).mean(0).squeeze(0))

    return torch.stack(per_frame_embeds, 0)


def _encode_texts(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Аналогично: text_model -> pooler_output -> text_projection.
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_emb = model.text_projection(text_out.pooler_output)
    return F.normalize(text_emb, dim=-1)


def _topk_pool_similarity(
    frame_embeds: torch.Tensor,   # (F, D)
    text_embeds: torch.Tensor,    # (P, D)
    model: CLIPModel,
    k: int = 4,
) -> torch.Tensor:
    sim = frame_embeds @ text_embeds.T
    sim = sim * model.logit_scale.exp()
    k = min(k, sim.size(0))
    topk_vals, _ = torch.topk(sim, k=k, dim=0)
    prompt_logits = topk_vals.mean(dim=0)
    return prompt_logits.softmax(dim=-1)


def classify_video(path: str | Path, top_k: int = 4) -> dict[str, str | float]:
    """
    Определяет тип гитары по видео с top-k агрегацией кадров,
    TTA и softmax-калибровкой по промптам.
    """
    frames = _extract_frames(path, num_frames=12)
    model, processor = _get_model_and_processor()

    # 1. Визуальные эмбеддинги кадров (с TTA, только "orig" + "flip")
    frame_embeds = _encode_frames(frames, model, processor, use_tta=True)

    # 2. Текстовые эмбеддинги всех промптов
    text_embeds = _encode_texts(ALL_PROMPTS, model, processor)

    # 3. Top-k агрегация + softmax по промптам.
    #    Возвращает вероятности (P,), сумма = 1.
    prompt_probs = _topk_pool_similarity(frame_embeds, text_embeds, model, k=top_k)

    # 4. Суммируем вероятности по классам.
    #    Так как сумма prompt_probs == 1, сумма по классам тоже == 1,
    #    а confidence == вероятность лучшего класса.
    class_scores: dict[str, float] = {}
    for prompt_idx, cls in enumerate(PROMPT_TO_CLASS):
        class_scores[cls] = class_scores.get(cls, 0.0) + float(prompt_probs[prompt_idx])

    best_class = max(class_scores, key=class_scores.get)
    total = sum(class_scores.values())
    confidence = class_scores[best_class] / total if total > 0 else 0.0

    return {
        "type": best_class,
        "confidence": round(float(confidence), 2),
    }


# =====================================================================
# Локальная проверка (без изменений в логике сверки)
# =====================================================================

def _normalize_name(name: str) -> str:
    name = str(name).strip().replace("\\", "/").split("/")[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    name = name.strip()
    if name.endswith("_unknown"):
        name = name[: -len("_unknown")]
    return name.strip()


if __name__ == "__main__":
    folder_path = Path("videos_for_classify")
    files_list = [f for f in folder_path.iterdir() if f.is_file()]

    print("Список полученных файлов для обработки:")
    print([str(f) for f in files_list])

    results: dict[str, str] = {}

    for one_video in files_list:
        try:
            prediction = classify_video(one_video, top_k=4)
            label = str(prediction["type"])
            conf = float(prediction["confidence"])

            results[one_video.name] = label
            print(f"\nРезультат анализа видео '{one_video}':")
            print(f"Определен класс: {label} (уверенность: {conf * 100:.0f}%)")
        except Exception as e:
            print(f"\nОшибка обработки '{one_video}': {e}")

    file_for_check = "guitar_type.txt"
    data_for_check: dict[str, str] = {}

    try:
        with open(file_for_check, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line.strip() or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                name = _normalize_name(parts[0])
                verdict = parts[-1].strip()
                if name and verdict:
                    data_for_check[name] = verdict
    except FileNotFoundError:
        print(f"\nФайл с эталонами '{file_for_check}' не найден.")

    if data_for_check:
        success = errors = missing = 0
        for name_song, verdict in results.items():
            key = _normalize_name(name_song)
            if key in data_for_check:
                expected = data_for_check[key]
                if verdict == expected:
                    success += 1
                    print(f"[OK] {key}: {verdict}")
                else:
                    errors += 1
                    print(f"[ERR] {key}: предсказано '{verdict}', ожидалось '{expected}'")
            else:
                missing += 1
                print(f"[SKIP] Нет эталона для: {key}")

        print(f"\nИтого: успешно={success}, ошибок={errors}, пропущено={missing}")
