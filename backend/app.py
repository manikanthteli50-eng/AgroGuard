"""
AgroGuard AI - Flask Backend API
Crop Disease Detection System
Production-ready with error handling and high-traffic support
"""

import os
import json
import uuid
import importlib
import requests
import numpy as np
import cv2
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from threading import Lock
from pathlib import Path
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Load environment variables from backend/.env before importing modules that read them
load_dotenv(Path(__file__).resolve().parent / ".env")

import database as db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model" / "crop_disease_model.h5")
CLASS_NAMES_PATH = str(BASE_DIR / "model" / "class_names.json")

model = None
class_names = []
model_lock = Lock()  # Thread-safe model access

try:
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model file not found at " + MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

try:
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            raw = json.load(f)
        class_names = [k for k, v in sorted(raw.items(), key=lambda x: x[1])]
        logger.info(f"{len(class_names)} classes loaded")
    else:
        logger.warning("Class names file not found")
except Exception as e:
    logger.error(f"Error loading class names: {str(e)}")

# Treatment Database
TREATMENTS = {
    "Tomato_Early_blight": {
        "description": "Caused by Alternaria solani fungus. Affects leaves, stems and fruit.",
        "symptoms": "Dark brown spots with concentric rings, yellowing around spots.",
        "treatment": [
            "Apply copper-based fungicide every 7–10 days.",
            "Remove and destroy infected leaves immediately.",
            "Avoid overhead watering; water at the base.",
            "Ensure good air circulation between plants.",
        ],
        "severity": "Moderate",
    },
    "Tomato_Late_blight": {
        "description": "Caused by Phytophthora infestans. Spreads rapidly in cool, wet weather.",
        "symptoms": "Water-soaked dark lesions on leaves and stems; white mold under leaves.",
        "treatment": [
            "Apply chlorothalonil or mancozeb fungicide immediately.",
            "Remove all infected plant material and burn it.",
            "Avoid planting tomatoes near potatoes.",
            "Use resistant varieties in future planting.",
        ],
        "severity": "High",
    },
    "Tomato_healthy": {
        "description": "Plant appears healthy with no visible disease signs.",
        "symptoms": "No symptoms detected.",
        "treatment": [
            "Continue regular watering and fertilization.",
            "Monitor regularly for early signs of disease.",
            "Maintain good soil drainage.",
        ],
        "severity": "None",
    },
    "Potato_Early_blight": {
        "description": "Fungal disease caused by Alternaria solani.",
        "symptoms": "Brown circular spots with concentric rings on lower leaves.",
        "treatment": [
            "Apply azoxystrobin or chlorothalonil fungicide.",
            "Remove lower infected leaves.",
            "Ensure adequate potassium fertilization.",
            "Practice crop rotation.",
        ],
        "severity": "Moderate",
    },
    "Potato_Late_blight": {
        "description": "Caused by Phytophthora infestans. Historically devastating.",
        "symptoms": "Dark, water-soaked spots; white fungal growth on leaf undersides.",
        "treatment": [
            "Apply metalaxyl-based fungicide immediately.",
            "Destroy all infected plants to prevent spread.",
            "Hill up soil around plants to protect tubers.",
            "Harvest tubers early if disease is severe.",
        ],
        "severity": "Very High",
    },
    "Potato_healthy": {
        "description": "Potato plant is healthy.",
        "symptoms": "No symptoms detected.",
        "treatment": ["Continue regular care and monitoring."],
        "severity": "None",
    },
    "Pepper_bell_Bacterial_spot": {
        "description": "Caused by Xanthomonas campestris bacteria.",
        "symptoms": "Small water-soaked spots that turn brown with yellow halos.",
        "treatment": [
            "Apply copper bactericide spray.",
            "Avoid working in field when plants are wet.",
            "Use disease-free certified seeds.",
            "Remove and destroy heavily infected plants.",
        ],
        "severity": "Moderate",
    },
    "Pepper_bell_healthy": {
        "description": "Pepper plant is healthy.",
        "symptoms": "No symptoms detected.",
        "treatment": ["Continue regular care and monitoring."],
        "severity": "None",
    },
    "Tomato_Bacterial_spot": {
        "description": "Caused by Xanthomonas bacteria.",
        "symptoms": "Small dark spots with yellow halos on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide.",
            "Remove infected plant debris.",
            "Avoid overhead irrigation.",
            "Use disease-free seeds.",
        ],
        "severity": "Moderate",
    },
    "Tomato_Leaf_Mold": {
        "description": "Caused by Passalora fulva fungus.",
        "symptoms": "Pale green to yellow spots on upper leaf surface; olive-green mold below.",
        "treatment": [
            "Improve air circulation and reduce humidity.",
            "Apply fungicide containing chlorothalonil.",
            "Remove infected leaves.",
            "Avoid wetting foliage when watering.",
        ],
        "severity": "Moderate",
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Caused by Septoria lycopersici fungus.",
        "symptoms": "Small circular spots with dark borders and gray centers on leaves.",
        "treatment": [
            "Apply fungicide with chlorothalonil or copper.",
            "Remove lower infected leaves.",
            "Mulch around plants to prevent soil splash.",
            "Practice crop rotation.",
        ],
        "severity": "Moderate",
    },
    "Tomato_Target_Spot": {
        "description": "Caused by Corynespora cassiicola fungus.",
        "symptoms": "Brown spots with concentric rings (target pattern) on leaves.",
        "treatment": [
            "Apply fungicide containing azoxystrobin.",
            "Remove infected plant material.",
            "Ensure good air circulation.",
            "Avoid overhead watering.",
        ],
        "severity": "Moderate",
    },
    "Tomato_Mosaic_virus": {
        "description": "Viral disease causing mosaic patterns on leaves.",
        "symptoms": "Mottled light and dark green patterns on leaves; stunted growth.",
        "treatment": [
            "No cure - remove and destroy infected plants.",
            "Control aphids and other vectors.",
            "Use virus-resistant varieties.",
            "Disinfect tools between plants.",
        ],
        "severity": "High",
    },
    "Tomato_YellowLeaf_Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies.",
        "symptoms": "Upward curling and yellowing of leaves; stunted plant growth.",
        "treatment": [
            "Control whitefly populations with insecticides.",
            "Remove and destroy infected plants.",
            "Use yellow sticky traps.",
            "Plant resistant varieties.",
        ],
        "severity": "Very High",
    },
    "Tomato_Spider_mites": {
        "description": "Tiny arachnids that feed on plant sap.",
        "symptoms": "Yellow stippling on leaves; fine webbing; leaf bronzing.",
        "treatment": [
            "Spray with miticide or insecticidal soap.",
            "Increase humidity around plants.",
            "Remove heavily infested leaves.",
            "Use predatory mites for biological control.",
        ],
        "severity": "Moderate",
    },
}

DEFAULT_TREATMENT = {
    "description": "Disease identified. Please consult a local agronomist.",
    "symptoms": "Refer to identified disease name for details.",
    "treatment": ["Consult your local agricultural extension office for guidance."],
    "severity": "Unknown",
}

# Flask App - Production Configuration
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["JSON_SORT_KEYS"] = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "jfif"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Weather configuration. Open-Meteo is keyless; OPENWEATHER_API_KEY is kept for
# optional OpenWeatherMap use. All values can be overridden via backend/.env.
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
GEOCODING_API_URL = os.environ.get("GEOCODING_API_URL", "https://geocoding-api.open-meteo.com/v1/search")
WEATHER_API_URL = os.environ.get("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_current_user():
    """Resolve the authenticated user from the Bearer token, or None."""
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
    if not token:
        return None
    try:
        return db.users().find_one({"token": token})
    except PyMongoError as e:
        logger.error("Token lookup failed: %s", e)
        return None


DETECTION_FIELDS = (
    "crop", "disease", "confidence", "severity", "status", "description",
    "symptoms", "treatment", "prevention", "weather", "weatherAdvice", "imageDataUrl",
)


def sanitize_detection(payload):
    """Keep only known detection fields from the client payload."""
    return {key: payload.get(key) for key in DETECTION_FIELDS if key in payload}


def preprocess_image(img_path):
    """Preprocess image with error handling"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        return np.reshape(img, (1, 128, 128, 3))
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def get_weather(city: str):
    """Get weather data from Open-Meteo (free, no API key needed)"""
    try:
        # First, get coordinates from city name
        geocoding_url = f"{GEOCODING_API_URL}?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return {
                "city": city,
                "temperature": "N/A",
                "humidity": "N/A",
                "wind_speed": "N/A",
                "description": "City not found",
                "icon": "01d",
            }
        
        location = geo_data['results'][0]
        latitude = location['latitude']
        longitude = location['longitude']
        
        # Get weather data
        weather_url = f"{WEATHER_API_URL}?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        current = weather_data['current']
        
        # Map weather codes to descriptions
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Foggy",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with hail",
        }
        
        description = weather_codes.get(current['weather_code'], "Unknown")
        
        return {
            "city": f"{location['name']}, {location.get('country', '')}",
            "temperature": round(current['temperature_2m'], 1),
            "humidity": current['relative_humidity_2m'],
            "wind_speed": round(current['wind_speed_10m'], 1),
            "description": description,
            "icon": "01d",
        }
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {
            "city": city,
            "temperature": "N/A",
            "humidity": "N/A",
            "wind_speed": "N/A",
            "description": "Could not fetch weather data",
            "icon": "01d",
        }


def predict_disease(img_path):
    """Predict disease from image with error handling"""
    try:
        if model is None:
            logger.warning("Model not loaded, returning default prediction")
            return "Tomato_Early_blight", 87.5

        with model_lock:
            img_array = preprocess_image(img_path)
            predictions = model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx]) * 100

            if class_names:
                label = class_names[top_idx]
            else:
                label = f"Class_{top_idx}"

            logger.info(f"Prediction: {label} ({confidence:.2f}%)")
            return label, round(confidence, 2)
    except Exception as e:
        logger.error(f"Error in predict_disease: {str(e)}")
        return "Unknown_Disease", 0.0


def generate_weather_advice(weather, disease_label):
    advice = []
    try:
        temp = float(weather["temperature"])
        humidity = float(weather["humidity"])

        if humidity > 80:
            advice.append("High humidity detected — fungal disease risk is elevated.")
        if temp > 30:
            advice.append("High temperature — ensure adequate irrigation.")
        if temp < 10:
            advice.append("Cold conditions — plant immunity may be lowered.")
        if "blight" in disease_label.lower() and humidity > 70:
            advice.append("Blight spreads fast in humid conditions. Act immediately.")
    except (TypeError, ValueError):
        pass

    if not advice:
        advice.append("Weather conditions appear favorable for crop growth.")
    return advice


# API Routes
@app.route("/", methods=["GET"])
def index():
    """Root route - shows available API endpoints instead of a 404."""
    return jsonify({
        "message": "AgroGuard AI Backend is running",
        "endpoints": {
            "GET /api/health": "Health check",
            "POST /api/predict": "Predict crop disease from an uploaded image",
        },
        "model_loaded": model is not None,
        "classes": len(class_names),
    })


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Return 204 so browsers don't log a 404 for the favicon."""
    return "", 204


@app.route("/api/health", methods=["GET"])
def health():
    db_ok = db.ping()
    return jsonify({
        "status": "healthy" if db_ok else "degraded",
        "model_loaded": model is not None,
        "classes": len(class_names),
        "database": "connected" if db_ok else "unreachable",
    })


@app.route("/api/register", methods=["POST"])
def register():
    """Create a new user account."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    try:
        if db.users().find_one({"email": email}):
            return jsonify({"error": "An account with this email already exists"}), 409
        token = uuid.uuid4().hex
        doc = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "token": token,
            "createdAt": db.utcnow_iso(),
        }
        db.users().insert_one(doc)
        logger.info("Registered new user: %s", email)
        return jsonify({"token": token, "user": db.public_user(doc)}), 201
    except PyMongoError as e:
        logger.error("Registration failed: %s", e)
        return jsonify({"error": "Database unavailable. Is MongoDB running on localhost:27017?"}), 503


@app.route("/api/login", methods=["POST"])
def login():
    """Authenticate a user and return a fresh token."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        user = db.users().find_one({"email": email})
        if not user or not check_password_hash(user.get("password", ""), password):
            return jsonify({"error": "Invalid email or password"}), 401
        token = uuid.uuid4().hex
        db.users().update_one({"_id": user["_id"]}, {"$set": {"token": token}})
        logger.info("User logged in: %s", email)
        return jsonify({"token": token, "user": db.public_user(user)}), 200
    except PyMongoError as e:
        logger.error("Login failed: %s", e)
        return jsonify({"error": "Database unavailable. Is MongoDB running on localhost:27017?"}), 503


@app.route("/api/me", methods=["GET"])
def me():
    """Return the currently authenticated user (used to restore a session)."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"user": db.public_user(user)}), 200


@app.route("/api/detections", methods=["GET"])
def list_detections():
    """List the authenticated user's saved detections (newest first)."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        cursor = db.detections().find({"email": user["email"]}).sort("createdAt", -1).limit(100)
        return jsonify({"detections": [db.serialize(d) for d in cursor]}), 200
    except PyMongoError as e:
        logger.error("List detections failed: %s", e)
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/detections", methods=["POST"])
def create_detection():
    """Save a detection record for the authenticated user."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    payload = sanitize_detection(request.get_json(silent=True) or {})
    if not payload.get("disease"):
        return jsonify({"error": "A disease label is required"}), 400
    try:
        now = db.utcnow_iso()
        doc = {**payload, "email": user["email"], "timestamp": now, "createdAt": now}
        result = db.detections().insert_one(doc)
        doc["_id"] = result.inserted_id
        # Enforce the 100-record cap per user
        old = list(db.detections().find({"email": user["email"]}).sort("createdAt", -1).skip(100))
        if old:
            db.detections().delete_many({"_id": {"$in": [d["_id"] for d in old]}})
        return jsonify({"detection": db.serialize(doc)}), 201
    except PyMongoError as e:
        logger.error("Save detection failed: %s", e)
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/detections", methods=["DELETE"])
def clear_detections():
    """Delete all detections for the authenticated user."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        result = db.detections().delete_many({"email": user["email"]})
        return jsonify({"deleted": result.deleted_count}), 200
    except PyMongoError as e:
        logger.error("Clear detections failed: %s", e)
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/profile", methods=["PATCH"])
def update_profile():
    """Update the authenticated user's display name."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "A valid name is required"}), 400
    try:
        db.users().update_one({"_id": user["_id"]}, {"$set": {"name": name}})
        updated = db.users().find_one({"_id": user["_id"]})
        return jsonify({"user": db.public_user(updated)}), 200
    except PyMongoError as e:
        logger.error("Profile update failed: %s", e)
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict disease from uploaded image"""
    try:
        if "image" not in request.files:
            logger.warning("No image file provided in request")
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        city = request.form.get("city", "New Delhi")

        if file.filename == "" or not allowed_file(file.filename):
            logger.warning(f"Invalid file: {file.filename}")
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, webp"}), 400

        # Save uploaded file
        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({"error": "Failed to save image"}), 500

        # Predict disease
        try:
            disease_label, confidence = predict_disease(filepath)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Failed to process image"}), 500
        finally:
            # Cleanup uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {str(e)}")

        # Get treatment info
        treatment_info = TREATMENTS.get(disease_label, DEFAULT_TREATMENT)
        
        # Get weather data
        try:
            weather = get_weather(city)
        except Exception as e:
            logger.warning(f"Weather fetch error: {str(e)}")
            weather = {"error": "Could not fetch weather"}
        
        # Generate advice
        try:
            weather_advice = generate_weather_advice(weather, disease_label)
        except Exception as e:
            logger.warning(f"Weather advice error: {str(e)}")
            weather_advice = []

        return jsonify({
            "disease": disease_label,
            "confidence": confidence,
            "severity": treatment_info.get("severity", "Unknown"),
            "description": treatment_info.get("description", ""),
            "symptoms": treatment_info.get("symptoms", ""),
            "treatment": treatment_info.get("treatment", []),
            "weather": weather,
            "weather_advice": weather_advice,
            "image_path": filename
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    logger.info("Starting AgroGuard AI Backend")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Classes available: {len(class_names)}")
    logger.info(f"MongoDB database: {db.DB_NAME}")
    # Use production WSGI server in deployment (gunicorn, waitress, etc.)
    app.run(debug=debug, host=host, port=port, threaded=True)
