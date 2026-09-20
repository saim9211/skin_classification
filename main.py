from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

import io
import os 

from src.predict import load_model, predict_image

app=FastAPI()

MODEL_PATH = os.path.join("models", "pneumonia_classifier.pth")

#load model one when api start 
model,class_names=load_model(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Classification API!"}
@app.post("/health")
def healthy_skin():
    """Endpoint to check if the skin is healthy."""
    return {"message": "The skin is healthy."}

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    """Endpoint to upload an image for prediction."""
    # Read the uploaded file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error occurred while processing the image.")

    # Make a prediction using the loaded model
    predicted_label, confidence_value, probability_list = predict_image(model, image, class_names)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predicted_label": predicted_label,
        "confidence_value": confidence_value,
        "probability_list":  {
            class_name: round(float(probability), 4)
            for class_name, probability in zip(
                class_names,
                probability_list
            )
        }
    }