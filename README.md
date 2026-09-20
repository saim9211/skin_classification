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
# FastApi
Basic Upload flow:
Image
 ↓
POST /predict
 ↓
FastAPI
 ↓
UploadFile
 ↓
PIL
 ↓
EfficientNet-B0
 ↓
Prediction
 ↓
JSON response

After the Validation:

             POST /predict
                   ↓
              UploadFile
                   ↓
           Check content_type
                   ↓
          JPEG / PNG ?
             ↙       ↘
           No        Yes
           ↓          ↓
         400       Read bytes
                       ↓
                 Empty file?
                   ↙    ↘
                 Yes     No
                 ↓        ↓
                400    PIL Image
                           ↓
                    Valid image?
                       ↙    ↘
                     No      Yes
                     ↓        ↓
                    400    EfficientNet
                              ↓
                         Prediction
                              ↓
                           JSON