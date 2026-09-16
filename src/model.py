import torch 
from torch import device, nn
from torchvision import models
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
model=torchvision.models.efficientnet_b0(weights=weights)

import torch.nn as nn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
out_dt = 2

for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-1].parameters():
    param.requires_grad = True
# Update the classifier head to suit our problem.
# For EfficientNet_B0, the classifier is a Sequential module at `model.classifier`.
model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(in_features=1280,  # EfficientNet_B0's last feature layer output size
              out_features=out_dt, # Use the dynamically determined number of output classes
              bias=True).to(device)
)

# Move the newly created classifier (part of current_model_instance) to the device.
model.classifier

