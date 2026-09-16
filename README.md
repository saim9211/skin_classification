# skin_classification
skin classification model


train class: do 
Dataset
   ↓
DataLoader
   ↓
EfficientNet-B0
   ↓
Training
   ↓
Validation
   ↓
Save .pth

evaluate do:

evaluate.py
     ↓
loads skin_classifier.pth
     ↓
test dataset
     ↓
predictions
     ↓
Accuracy
Precision
Recall
F1
Classification Report
Confusion Matrix


final project 

                    NOTEBOOK
                       │
                       │ experimentation
                       ↓
              ┌──────────────────┐
              │ Final EfficientNet│
              │       B0         │
              └────────┬─────────┘
                       │
                       ↓
                 skin_classifier.pth
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       train.py   evaluate.py   predict.py
          │            │            │
       Training     Evaluation    Inference