import cv2
from deepface import DeepFace
import pandas as pd
import datetime

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not detected.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")

    log_file = "emotion_log.csv"
    pd.DataFrame(columns=["Time", "Emotion"]).to_csv(log_file, index=False)

    print("[INFO] Starting realtime emotion recognition. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) == 0:
            cv2.putText(frame, "No face", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if isinstance(result, list):
                    emotion = result[0].get('dominant_emotion', 'No face')
                else:
                    emotion = result.get('dominant_emotion', 'No face')

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pd.DataFrame([[timestamp, emotion]], columns=["Time", "Emotion"]) \
                  .to_csv(log_file, mode='a', header=False, index=False)

            except Exception as e:
                print(f"[WARNING] DeepFace failed on frame: {e}")

        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()