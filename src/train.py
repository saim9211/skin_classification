import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# Paths
# =========================

train_dir = "data/train"
val_dir = "data/val"

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)


# =========================
# Transforms
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Dataset
# =========================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# DataLoader
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# =========================
# Number of classes
# =========================

num_classes = len(train_dataset.classes)


# =========================
# Model
# =========================

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, num_classes)
)

model = model.to(device)


# =========================
# Loss and Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# =========================
# Training
# =========================

epochs = 5

for epoch in range(epochs):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total


    # =========================
    # Validation
    # =========================

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


# =========================
# Save Model
# =========================

model_path = os.path.join(
    model_dir,
    "skin_classifier.pth"
)

torch.save({
    "model_name": "efficientnet_b0",
    "num_classes": num_classes,
    "class_names": train_dataset.classes,
    "model_state_dict": model.state_dict()
}, model_path)

print("\nModel saved at:", model_path)