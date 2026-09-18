import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Shared transform
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
# Model loader
# =========================

def load_model(model_path):
    """Load a trained EfficientNet-B0 checkpoint and return the model + class names."""
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError("Checkpoint does not contain a model state dict.")

        num_classes = checkpoint.get("num_classes")
        class_names = checkpoint.get("class_names")
    else:
        raise TypeError("Unsupported checkpoint format.")

    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            raise ValueError("Checkpoint missing num_classes and class_names.")

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, class_names


# =========================
# Prediction helper
# =========================

def predict_image(model, image, class_names):
    """Return predicted label, confidence, and class probabilities for a PIL image."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a file path or a PIL.Image object")

    image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_index = int(predicted_index.item())
    predicted_label = class_names[predicted_index]
    confidence_value = float(confidence.item())
    probability_list = [float(p) for p in probabilities.tolist()]

    return predicted_label, confidence_value, probability_list


# =========================
# Optional direct run for testing
# =========================

if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join("models", "pneumonia_classifier.pth"))
    image_path = os.path.abspath(os.path.join("predict_Data", "random_check.jpeg"))

    model, class_names = load_model(model_path)
    predicted_label, confidence, probabilities = predict_image(model, image_path, class_names)

    print("Prediction:", predicted_label)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("Probabilities:", probabilities)