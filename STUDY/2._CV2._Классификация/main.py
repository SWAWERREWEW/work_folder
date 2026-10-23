import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


# ----------------------------
# Константы
# ----------------------------
DATA_DIR = Path(".")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test_new"
CLASS_CSV = DATA_DIR / "class_names.csv"

IMG_SIZE = 160
BATCH_SIZE = 64
EPOCHS = 12
LR = 3e-4
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NUM_WORKERS = 2  # если снова "worker exited unexpectedly" — поставьте 0


# ----------------------------
# Dataset
# ----------------------------
class AnimalDataset(Dataset):
    def __init__(self, root, class_to_idx=None, transform=None, paths=None):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform

        if paths is None:
            self.paths = sorted([
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ])
        else:
            self.paths = list(paths)

        if len(self.paths) == 0:
            raise RuntimeError(f"Не найдено изображений в {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.class_to_idx is not None:
            cls_name = path.stem.split("_")[0]
            label = self.class_to_idx[cls_name]
            return image, label
        return image, path.name


# ----------------------------
# Модель
# ----------------------------
def make_model(num_classes):
    try:
        from torchvision.models import ResNet18_Weights
        model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print("Используются pretrained-веса ResNet18")
    except Exception as e:
        print("Pretrained-веса недоступны, обучение с нуля:", e)
        model = torchvision.models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ----------------------------
# class_names.csv
# ----------------------------

def ensure_class_names_csv():
    # Если файл уже есть (предоставлен платформой) - НЕ трогаем его!
    if CLASS_CSV.exists():
        print(f"Используем существующий {CLASS_CSV}")
        return

    # Если файла нет, пытаемся собрать из train (fallback)
    print(f"{CLASS_CSV} не найден, генерируем из имен файлов...")
    names = sorted({
        p.stem.split("_")[0] # Убедитесь, что это работает для ваших файлов
        for p in TRAIN_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    })
    # БЕЗ пробелов в ключах и имени файла!
    df = pd.DataFrame({"name": names, "label": range(len(names))})
    df.to_csv(CLASS_CSV, index=False) # Убрали пробел в имени файла
    print("Создан class_names.csv")

# def ensure_class_names_csv():
#     correct_order = ['panda', 'bear', 'lion', 'deer', 'tiger', 'bird', 'horse',
#                      'dolphin', 'elephant', 'dog', 'zebra', 'giraffe', 'kangaroo', 'cat']
#     df = pd.DataFrame({"name": correct_order, "label": range(len(correct_order))})
#     df.to_csv("class_names.csv", index=False)
#     print("Создан правильный class_names.csv")
    # if CLASS_CSV.exists():
    #     print(f"Используем существующий {CLASS_CSV}")
    #     return
    #
    # print(f"{CLASS_CSV} не найден, генерируем из имён файлов train")
    # names = sorted({
    #     p.stem.split("_")[0]
    #     for p in TRAIN_DIR.rglob("*")
    #     if p.is_file() and p.suffix.lower() in IMG_EXTS
    # })
    # df = pd.DataFrame({"name": names, "label": range(len(names))})
    # df.to_csv(CLASS_CSV, index=False)
    # print(df)


# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    ensure_class_names_csv()

    classes_df = pd.read_csv(CLASS_CSV)
    class_to_idx = {
        str(row["name"]): int(row["label"])
        for _, row in classes_df.iterrows()
    }
    num_classes = len(class_to_idx)
    print("num_classes =", num_classes)
    print(class_to_idx)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Один раз собираем пути и делим на train/val
    full = AnimalDataset(TRAIN_DIR, class_to_idx=None, transform=None)
    all_paths = list(full.paths)
    random.shuffle(all_paths)

    val_size = int(0.15 * len(all_paths))
    val_paths = all_paths[:val_size]
    train_paths = all_paths[val_size:]

    # Проверим, что все классы из train есть в class_to_idx
    missing = {p.stem.split("_")[0] for p in all_paths} - set(class_to_idx)
    if missing:
        raise RuntimeError(f"В class_names.csv нет классов: {missing}")

    train_ds = AnimalDataset(TRAIN_DIR, class_to_idx, train_tf, paths=train_paths)
    val_ds = AnimalDataset(TRAIN_DIR, class_to_idx, val_tf, paths=val_paths)

    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )

    model = make_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_path = "best_model.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  saved best model, val_acc={best_acc:.4f}")

    print("Best val_acc =", best_acc)

    # Предсказание
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_ds = AnimalDataset(TEST_DIR, class_to_idx=None, transform=val_tf)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )

    rows = []
    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            for file_name, pred in zip(file_names, preds):
                rows.append((file_name, int(pred)))

    submission = pd.DataFrame(rows, columns=["FILE_NAME", "TARGET"])
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("submission.csv сохранён, строк:", len(submission))
    result = pd.read_csv("submission.csv")
    print(result.head(15))


if __name__ == "__main__":
    main()
