# Chest X-Ray Pneumonia Classification

A portfolio-style deep learning project for chest X-ray image classification using EfficientNet-B0, trained to distinguish between normal and pneumonia-affected X-rays. The project includes model training, evaluation, prediction utilities, a FastAPI backend, and a Streamlit frontend for an interactive demo.

Link: https://huggingface.co/spaces/saim9211/skin_classification

## Project Overview

This project demonstrates a complete machine learning pipeline for medical image classification:

- Data preparation and preprocessing
- CNN model training with PyTorch
- Validation and evaluation metrics
- Model checkpoint saving
- Prediction pipeline for new images
- REST API deployment with FastAPI
- Web demo with Streamlit

The goal is to build a clean, real-world classification workflow with a strong emphasis on clarity, usability, and presentation suitable for a portfolio.

## Problem Statement

Chest X-ray imaging is one of the most common diagnostic tools in medical practice. Pneumonia is a serious condition that can be challenging to identify quickly and consistently from imaging alone. This project aims to automate the diagnosis support process by developing a deep learning model that classifies chest X-ray images into the following classes:

- NORMAL
- PNEUMONIA

## Model Architecture

The solution uses EfficientNet-B0, a lightweight yet highly effective convolutional neural network known for strong performance on image classification tasks while maintaining good efficiency.

### Why EfficientNet-B0?

- Strong feature extraction capability
- Good balance between accuracy and computational cost
- Widely used for transfer learning and medical imaging tasks
- Performs well with moderate-size datasets

## Dataset

The project uses a chest X-ray dataset organized into train and test folders, with images separated by class labels. The dataset structure is designed to work with PyTorch ImageFolder, which automatically handles class labels from directory naming.

Get the Dataset from Following Link: https://drive.google.com/file/d/1VwEw8HMu1XLQlCy2_2L5JrzaSmVDqqFO/view?usp=drive_link

## Project Structure

```text
skin_classification/
├── app.py                     # Streamlit frontend
├── main.py                   # FastAPI prediction API
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── models/
│   └── pneumonia_classifier.pth
├── data/
│   └── chest_xray_1/
│       └── chest_xray/
│           ├── train/
│           └── test/
├── src/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation and metrics
│   ├── predict.py           # Prediction helpers and model loader
│   └── model.py             # Model definition / experimentation
├── predict_Data/
│   └── random_check.jpeg     # Sample prediction image
└── skin_image_classification_model.ipynb
```

## Workflow

### 1. Model Training
The training pipeline loads the chest X-ray dataset, applies preprocessing transforms, builds an EfficientNet-B0 model, and trains it to classify pneumonia versus normal cases.

### 2. Model Saving
Once training is complete, the model is saved as a checkpoint containing:

- model state dictionary
- number of classes
- class names
- training metadata

### 3. Evaluation
The evaluation script loads the saved model, runs it on the test set, and calculates metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification report
- Confusion matrix

### 4. Prediction
The prediction pipeline loads the trained model and performs inference on uploaded images. It also includes a confidence threshold to reject uncertain predictions and return `Unknown` when confidence is below 50%.

### 5. API Integration
FastAPI exposes a prediction endpoint that accepts an uploaded image and returns:

- file name
- content type
- predicted label
- confidence score
- probability distribution for each class

### 6. Frontend Demo
A Streamlit app provides a polished dashboard-style user interface where a user can upload a chest X-ray image and receive the model’s output in real time.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- OpenCV / PIL
- FastAPI
- Streamlit
- NumPy
- Matplotlib
- Seaborn

## Model Behavior

The model outputs probabilities for each class and selects the class with the highest confidence. To improve reliability, predictions below 50% confidence are treated as `Unknown`, preventing false certainty in uncertain cases.

## API Endpoint

### Predict endpoint

```bash
POST /predict
```

Request:
- image file via multipart form upload

Response example:

```json
{
  "filename": "random_check.jpeg",
  "content_type": "image/jpeg",
  "predicted_label": "PNEUMONIA",
  "confidence_value": 0.9968,
  "probability_list": {
    "NORMAL": 0.0032,
    "PNEUMONIA": 0.9968
  }
}
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

### 4. Upload an image

Open the Streamlit app in the browser and upload a chest X-ray image to receive a prediction.

## Results and Evaluation

The model is evaluated using standard classification metrics to assess how well it distinguishes between healthy and pneumonia cases. These metrics help measure both accuracy and reliability in medical classification tasks.

## Project Goals

This project was created to showcase:

- end-to-end ML workflow development
- hospital-style image classification
- deployment-ready API design
- portfolio-grade user interface
- real-world engineering practices in model inference systems

## Notes

This application is intended for educational and research purposes. It should not be treated as a substitute for medical diagnosis or professional clinical decision-making.

## Future Improvements

Possible enhancements for future iterations include:

- adding more classes beyond normal and pneumonia
- using larger and more diverse datasets
- experimenting with transfer learning improvements
- adding model explainability tools
- deploying the API to cloud hosting
- improving the Streamlit experience with richer analytics

## Conclusion

This project demonstrates a complete deep learning solution for chest X-ray pneumonia classification, covering training, evaluation, deployment, and interactive demonstration. It is structured to be both technically strong and visually presentable as a portfolio project.
