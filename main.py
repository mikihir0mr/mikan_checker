import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Model Path
# Use relative path for compatibility with deployment environments (Render, etc.)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tflite_models-20251205T022915Z-3-001", "tflite_models", "mikan_classifier.tflite")

# Class Labels (Lexicographically sorted as per flow_from_directory default)
CLASS_NAMES = ['10', '11', '12', '13', '14', '15', '8', '9']

# Load TFLite Model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return {"error": "Model not loaded"}

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        input_data = np.array(image, dtype=np.float32)
        input_data = input_data / 255.0  # Normalize
        input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        predicted_index = np.argmax(output_data[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(output_data[0][predicted_index])

        return {
            "sugar_content": predicted_class,
            "confidence": confidence,
            "all_scores": {name: float(score) for name, score in zip(CLASS_NAMES, output_data[0])}
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
