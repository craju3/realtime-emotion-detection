from flask import Flask, render_template, Response
import cv2
from facenet_pytorch import MTCNN
from fer import FER
import threading
import tensorflow as tf

app = Flask(__name__)

# Check if TensorFlow is using GPU
print("TensorFlow is using GPU:", tf.config.list_physical_devices('GPU'))

# Initialize MTCNN for face detection
print("Initializing MTCNN...")
mtcnn = MTCNN()

# Initialize FER for emotion detection
print("Initializing FER...")
emotion_detector = FER(mtcnn=True)

# Initialize webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

frame = None
lock = threading.Lock()

def capture_frame():
    global frame
    frame_count = 0
    while True:
        ret, new_frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        if frame_count % 3 == 0:  # Process every third frame
            # Resize frame for faster processing
            new_frame = cv2.resize(new_frame, (640, 480))
            
            with lock:
                frame = new_frame

# Start a thread to capture frames from the webcam
print("Starting capture thread...")
capture_thread = threading.Thread(target=capture_frame)
capture_thread.daemon = True
capture_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            frame_copy = frame.copy()

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(frame_copy)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame_copy[y1:y2, x1:x2]

                # Detect emotion in the face
                emotions = emotion_detector.detect_emotions(face)
                if emotions:
                    emotion = emotions[0]['emotions']
                    emotion_label = max(emotion, key=emotion.get)
                    emotion_percentage = emotion[emotion_label] * 100

                    # Draw bounding box and emotion label with percentage
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{emotion_label}: {emotion_percentage:.2f}%"
                    cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, jpeg = cv2.imencode('.jpg', frame_copy)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
