from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import time
from PIL import Image

app = FastAPI()

# Load the trained model
model = load_model("resnet50_image_model.h5")

# Define class labels
CLASS_LABELS = ["Healthy", "Hypothyroidism", "Hyperthyroidism"]

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        start_time = time.time()  # Start time tracking

        # Read and preprocess the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # Resize for ResNet50
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_LABELS[predicted_index]
        confidence_score = float(np.max(prediction))  # Convert to float

        # Determine if it's a disease and specify the type
        if predicted_class == "Healthy":
            disease_status = "No thyroid disease detected"
        else:
            disease_status = f"Thyroid Disease Detected: {predicted_class}"

        # Top-3 Predictions (Optional)
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {"class": CLASS_LABELS[i], "confidence": float(prediction[0][i])}
            for i in top_3_indices
        ]

        end_time = time.time()  # End time tracking
        prediction_time = round(end_time - start_time, 3)  # Time in seconds

        return {
            "prediction": predicted_class,
            "disease_status": disease_status,
            "confidence": confidence_score,
            "top_3_predictions": top_3_predictions,
            "model": "ResNet50",
            "input_shape": img_array.shape,
            "prediction_time": f"{prediction_time} sec"
        }

    except Exception as e:
        return {"error": str(e)}
