# Real-Time Emotion Detection

This project is a real-time emotion detection application using Flask, OpenCV, MTCNN, and FER.

## Features

- **Real-time face detection using MTCNN**
- **Emotion detection using FER**
- **Web-based interface using Flask**

## Requirements

- Python 3.6+
- Flask
- OpenCV
- facenet-pytorch
- FER
- TensorFlow

## Setup Instructions

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/craju3/realtime-emotion-detection.git
    cd realtime-emotion-detection
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask application:

    ```bash
    python app.py
    ```

   The Flask application will start running locally at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Folder Structure

```plaintext
realtime-emotion-detection/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── venv/
