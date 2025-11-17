# Face Emotion Recognition App

## Overview
This project detects human emotions (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust) from facial expressions using the DeepFace library and OpenCV.  
It provides a simple Flask web interface where users can upload an image, and the app will analyze and display the dominant emotion.

## Features
- Image Upload: Upload any face image and get emotion analysis instantly.  
- Pre‑trained DeepFace Models: Uses state‑of‑the‑art emotion recognition models.  
- Face Detection: Haar Cascade ensures only face regions are analyzed.  
- Simple Web UI: Flask renders a clean interface for uploads and results.  
- CSV Logging (optional): Standalone script logs emotions with timestamps for later analysis.  

## Project Structure

FaceEmotionApp/
│
├── app.py                 # Flask app (upload + emotion analysis)
├── realtime_emotion.py    # Standalone script (webcam + CSV logging)
├── requirements.txt       # Dependencies
├── emotion_log.csv        # Auto‑created by realtime script
├── templates/
│   └── index.html         # Upload form
└── static/
    └── uploads/           # Uploaded images saved here

## Setup & Installation

1. Clone or download this repository
   git clone https://github.com/yourusername/FaceEmotionApp.git
   cd FaceEmotionApp
   
2. Create a virtual environment
   python -m venv venv
   # Mac/Linux
   source venv/bin/activate
   # Windows PowerShell
   .\venv\Scripts\activate
   
3. Install dependencies
   pip install -r requirements.txt
   
## How to Run

>> Run Flask Web App
python app.py
 - Open your browser at: http://127.0.0.1:5000  
 - Upload an image → see emotion result.

>> Run Standalone Realtime Script
python realtime_emotion.py
 - Opens webcam window.  
 - Press q to quit.  
 - Logs emotions into emotion_log.csv.  

## Sources / Libraries Used
- [DeepFace](https://github.com/serengil/deepface) → Pre‑trained emotion recognition models  
- [OpenCV](https://opencv.org/) → Face detection and image processing  
- [Flask](https://flask.palletsprojects.com/) → Web framework for UI  
- [Pandas](https://pandas.pydata.org/) → CSV logging  
- [NumPy](https://numpy.org/) → Array operations  

## Benefits
- Quick Emotion Insights: Detect emotions from facial expressions in seconds.  
- Educational Value: Learn how AI models can interpret human emotions.  
- Extensible: Can be integrated into larger projects (HR tools, customer feedback analysis, interactive apps).  
- Beginner‑Friendly: Clear structure and requirements make it easy to run and extend.  

## Most Important -- Notes
- Ensure good lighting and frontal face images for best accuracy.  
- Webcam feed was removed from Flask for simplicity; use realtime_emotion.py if you want live detection.
- DeepFace internally loads the emotion model — no need to pass model argument.  