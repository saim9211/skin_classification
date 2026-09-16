import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# Paths
# =========================

test_dir = "data/test"
model_path = "models/skin_classifier.pth"


# =========================
# Transform
# =========================

test_transform = transforms.Compose([
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

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# =========================
# Load Checkpoint
# =========================

checkpoint = torch.load(
    model_path,
    map_location=device
)

num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]


# =========================
# Model
# =========================

model = models.efficientnet_b0(
    weights=None
)

model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, num_classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# =========================
# Prediction
# =========================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\nEvaluation Results")
print("------------------")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# =========================
# Classification Report
# =========================

print("\nClassification Report")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Skin Disease Classification Confusion Matrix")

plt.tight_layout()

plt.show()