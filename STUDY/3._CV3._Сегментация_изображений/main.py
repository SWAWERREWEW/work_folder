import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# =====================================================================
# 1. ГИПЕРПАРАМЕТРЫ И НАСТРОЙКИ
# =====================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
IMG_SIZE = 128  # Размер, к которому приводим картинки для обучения

# Пути к вашей структуре папок
TRAIN_IMG_DIR = "./train/images/"
TRAIN_MASK_DIR = "./train/masks/"
TEST_IMG_DIR = "./test_64/images/"  # Изменено под test_64/images/ из вашего описания

# =====================================================================
# 2. ФУНКЦИЯ RLE-КОДИРОВАНИЯ (Строго по условию задачи)
# =====================================================================
def rle_encode(mask):
    """
    Преобразует бинарную маску (0 и 1) в строку формата Run-Length Encoding.
    Возвращает строку пар: 'start_index length start_index length ...'
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return ' '.join(str(x) for x in runs)

# =====================================================================
# 3. КАСТОМНЫЕ ДАТАСЕТЫ (Dataset)
# =====================================================================
class PetTrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Берем только файлы картинок
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Разделяем имя файла и расширение (например, 'boxer_179' и '.jpg')
        name_without_ext, _ = os.path.splitext(img_name)

        # Находим маску, проверяя разные расширения (.png, .jpg)
        mask_name = None
        for ext in ['.png', '.jpg', '.png', '.jpeg']:
            potential_mask = name_without_ext + ext
            if os.path.exists(os.path.join(self.mask_dir, potential_mask)):
                mask_name = potential_mask
                break

        if mask_name is None:
            raise FileNotFoundError(f"Не удалось найти маску для изображения {img_name} в папке {self.mask_dir}")

        mask_path = os.path.join(self.mask_dir, mask_name)

        # Читаем изображение
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Читаем маску
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Бинаризуем (делаем строго 0 и 1)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)


class PetTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]  # Запоминаем исходный размер для RLE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, img_name, orig_h, orig_w

if __name__ == '__main__':
    # --- БЛОК 4: Инициализация DataLoader ---
    train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),])

    test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
    ])


    train_dataset = PetTrainDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = PetTestDataset(TEST_IMG_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # батч=1 для простоты ресайза обратно
   
    # --- БЛОК 5: Инициализация экземпляра модели ---
    # Инициализируем Unet++ с бэкбоном EfficientNet-B1 согласно вашему шаблону
    model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",  # предобученные веса для энкодера разрешены
    in_channels=3,
    classes=1,                   # 1 класс на выходе (бинарная сегментация)
    ).to(DEVICE)

    # Комбинированная функция потерь (Dice Loss + BCE) идеально подходит для сегментации
    # criterion = smp.losses.DiceLoss(mode='binary') # или nn.BCEWithLogitsLoss()
    # criterion = smp.losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- БЛОК 6: Обучение ---
    print("Начало обучения модели...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
    
        for images, masks in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Средняя потеря (Loss) за эпоху: {epoch_loss / len(train_loader):.4f}")

    # =====================================================================
    # 7. ФОРМИРОВАНИЕ ПРЕДСКАЗАНИЙ И КОДИРОВАНИЕ В RLE
    # =====================================================================
    print("\nГенерация предсказаний для теста и кодирование в RLE...")
    model.eval()
    submission_data = []

    with torch.no_grad():
        for image, img_name, orig_h, orig_w in tqdm(test_loader, desc="Тестирование"):
            image = image.to(DEVICE)
            output = model(image)

            # Пропускаем через сигмоиду и берем порог 0.5
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            # ВАЖНО: возвращаем маску к ИСХОДНОМУ размеру картинки из теста
            orig_h, orig_w = orig_h.item(), orig_w.item()
            pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Кодируем маску в RLE строку
            rle_string = rle_encode(pred_mask_resized)

            # Если маска оказалась пустой, то по условию Dice=1.0, но в файл пишем пустую строку или результат RLE
            submission_data.append({
                "FILE_NAME": img_name[0],
                "TARGET": rle_string
            })

    # =====================================================================
    # 8. СОХРАНЕНИЕ CSV С ЗАГОЛОВКОМ (По требованиям на скриншоте)
    # =====================================================================
    df = pd.DataFrame(submission_data)
    # Сохраняем строго со столбцами FILE_NAME, TARGET
    df.to_csv("submission.csv", index=False)
    print("\nГотово! Файл 'submission.csv' успешно сформирован и готов к отправке.")
