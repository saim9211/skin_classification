import requests

API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "predict_Data/random_check.jpeg"

with open(IMAGE_PATH, "rb") as image_file:
    response = requests.post(
        API_URL,
        files={"file": ("random_check.jpeg", image_file, "image/jpeg")}
    )

print("Status Code:", response.status_code)

if response.status_code == 200:
    result = response.json()
    print("Filename:", result["filename"])
    print("Prediction:", result["predicted_label"])
    print("Confidence:", result["confidence_value"])
else:
    print("Error:", response.json())