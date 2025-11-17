import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, redirect, url_for
import cv2
from deepface import DeepFace

app = Flask(__name__, template_folder="templates", static_folder="static")

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")


def analyze_face_roi(bgr_img):
    """Convert BGR image to RGB and analyze emotion."""
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(
        img_path=rgb_img,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='opencv'
    )
    if isinstance(result, list):
        return result[0].get('dominant_emotion', 'No face')
    return result.get('dominant_emotion', 'No face')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(url_for('index'))

    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static", "uploads", file.filename)
    file.save(filepath)

    try:
        bgr = cv2.imread(filepath)
        if bgr is None:
            return "Error: Unable to read uploaded image."

        emotion = analyze_face_roi(bgr)
        img_url = url_for('static', filename='uploads/' + file.filename)

        return f"""
        <h2>Dominant emotion: {emotion}</h2>
        <img src="{img_url}" width="300">
        <br><a href="/">Back</a>
        """
    except Exception as e:
        return f"Error analyzing image: {e}"


if __name__ == '__main__':
    app.run(debug=False)