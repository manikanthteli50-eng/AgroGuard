# 🌾 AgroGuard AI - Smart Crop Disease Detection System

AI-powered crop disease detection using CNN with real-time weather integration.

## � Project Overview

Crop diseases are a major challenge in agriculture, causing 20–40% crop loss worldwide every year. AgroGuard AI provides an intelligent solution by combining:

- Deep Learning for plant disease detection
- Computer Vision for leaf image analysis
- Weather data analysis for disease risk prediction

## 📁 Project Structure

```
AI-BASED-CROP-DISEASE-DETECTION-SYSTEM/
├── frontend-react/        # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/               # Backend (Flask API)
│   ├── app.py
│   ├── train_model.py
│   ├── prepare_dataset.py
│   ├── evaluate_model.py
│   ├── requirements.txt
│   └── model/            # Trained model (generated)
│
├── archive/              # PlantVillage dataset (downloaded separately)
├── dataset/               # Training data (generated)
│   ├── train/
│   └── test/
└── README.md
```

## 🚀 Quick Start

From the project root, you can run the scripts without changing directories:

### Step 1: Prepare Dataset

```bash
python backend/prepare_dataset.py
```

This organizes images from the `archive/PlantVillage` folder into train/test splits under `dataset/`.

### Step 2: Train the Model

```bash
python backend/train_model.py
```

Training takes 15-30 minutes. The model is saved to `backend/model/`.

### Step 3: Start Backend Server

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend runs on `http://localhost:5000`

### Step 4: Open Frontend

Open `frontend-react/index.html` in your browser or start the React dev server:

```bash
cd frontend-react
npm install
npm run dev
```

Visit the local Vite URL shown in the terminal.

## 🔑 Weather API (Optional)

Get a free API key from [OpenWeatherMap](https://openweathermap.org/api):

```bash
export OPENWEATHER_API_KEY=your_key_here
```

## 🦠 Supported Diseases

- Pepper Bell: Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Tomato: Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Spider Mites, Healthy

## 🛠️ Tech Stack

- Frontend: HTML5, CSS3, Vanilla JavaScript
- Backend: Flask, TensorFlow/Keras
- Model: CNN (Convolutional Neural Network) - Production-optimized
- Dataset: PlantVillage
- API: OpenWeatherMap for real-time weather data

## 📝 API Endpoints

- `GET /api/health` - Check server status
- `POST /api/predict` - Analyze crop image

## ⭐ Key Features

- 🌿 AI Crop Disease Detection using CNN
- 📷 Image upload and analysis
- � Disease prediction with confidence scores
- 🌦️ Weather-based disease risk analysis
- 💊 Treatment recommendations
- 🔒 Production-ready error handling
- ⚡ High-traffic support with thread-safe model access

## 🌍 Impact

AgroGuard AI helps:

- Farmers detect diseases early
- Reduce crop losses
- Improve agricultural productivity
- Promote AI-based smart farming solutions

Built with ❤️ for farmers



Final year project