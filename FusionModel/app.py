from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load all models from Breakhis_Models folder
MODEL_DIR = "Breakhis_Models"
models = {}
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".h5"):
        model_name = model_file.replace(".h5", "")
        models[model_name] = load_model(os.path.join(MODEL_DIR, model_file))

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    # Process the image and get predictions from each model
    img_array = preprocess_image(file_path)
    predictions = {}

    for model_name, model in models.items():
        pred = model.predict(img_array)[0][0]
        label = "Malignant" if pred > 0.5 else "Benign"
        predictions[model_name] = {"Probability": float(pred), "Prediction": label}

    return jsonify({"image_url": file_path, "predictions": predictions})


FUSION_MODEL_PATH = 'Breakhis_Models/FusionModel.h5'
fusion_model = load_model(FUSION_MODEL_PATH)
@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    # Check if both files are present
    if "outside_image" not in request.files or "inside_image" not in request.files:
        return jsonify({"error": "Both outside and inside images are required"}), 400
    
    outside_file = request.files["outside_image"]
    inside_file = request.files["inside_image"]
    
    if outside_file.filename == "" or inside_file.filename == "":
        return jsonify({"error": "No file selected for one or both images"}), 400
    
    # Save both images
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    outside_path = os.path.join(upload_dir, outside_file.filename)
    inside_path = os.path.join(upload_dir, inside_file.filename)
    
    outside_file.save(outside_path)
    inside_file.save(inside_path)
    
    try:
        # Preprocess only the inside image for prediction
        img_array = preprocess_image(inside_path)
        
        # Get prediction from fusion model
        pred = fusion_model.predict(img_array)[0][0]
        label = "Malignant" if pred > 0.5 else "Benign"
        
        # Create response with both image URLs and prediction
        response = {
            "outside_image_url": outside_path,
            "inside_image_url": inside_path,
            "predictions": {
                "FusionModel": {
                    "Probability": float(pred),
                    "Prediction": label
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
