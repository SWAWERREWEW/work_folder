import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# 1. Настройки и гиперпараметры
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Папка, где лежат все картинки для обучения
TRAIN_DIR = "./dataset/train"
# Папка, где лежат все картинки для теста
TEST_DIR = "./dataset/test"
# Путь к файлу разметки со столбцами file_name,class
CSV_PATH = "./dataset/train.csv"

# Список классов строго в алфавитном порядке для правильной индексации (0-4)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

# 2. Трансформации изображений
train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Кастомный датасет для Обучающей выборки (читает CSV)
class TrainDatasetFromCSV(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Чтение CSV-файл разметки
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Получение имя файла и текстовый класс
        img_name = self.df.iloc[idx]['file_name']
        label_name = self.df.iloc[idx]['class']
        
        # Перевод текстовый класс в число (0, 1, 2, 3 или 4)
        label_idx = CLASS_TO_IDX[label_name]
        
        # Загрузка картинку
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# 4. Кастомный датасет для Тестовой выборки (все файлы из папки)
class TestDatasetFromFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Сортировка файлов, чтобы порядок предсказаний всегда был одинаковым
        self.all_imgs = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# Создание загрузчика данных
train_dataset = TrainDatasetFromCSV(csv_file=CSV_PATH, img_dir=TRAIN_DIR, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TestDatasetFromFolder(img_dir=TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Архитектура нейросети CNN (С нуля)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 16x16
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 8x8
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 5 классов
            nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Обучение модели
print("Начало обучения модели...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Эпоха {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# 7. Генерация финальных предсказаний для теста
print("\nЗапуск предсказаний на тестовом наборе...")
model.eval()
submission_results = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        for filename, pred_idx in zip(filenames, preds):
            pred_label = CLASS_NAMES[pred_idx.item()]
            submission_results.append([filename, pred_label])

# 8. Сохранение строго по правилам таска (без шапки, без индексов)
# Итоговый файл будет называться final_submission.csv
df_out = pd.DataFrame(submission_results)
df_out.to_csv("submission.csv", index=False, header=False, encoding='utf-8')
print("\nВсё готово! Файл 'submission.csv' успешно создан и готов к отправке.")
