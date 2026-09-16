import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Model Path
# =========================

model_path = "models/skin_classifier.pth"


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
# Image Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Image Path
# =========================

image_path = "test_image.jpg"

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# =========================
# Prediction
# =========================

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


# =========================
# Result
# =========================

predicted_class = predicted_class.item()

confidence = confidence.item()

print("Prediction:", class_names[predicted_class])

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)