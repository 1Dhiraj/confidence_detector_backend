from flask import Flask, Response, request, json
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import logging
from flask_cors import CORS
import base64
import os
from io import BytesIO
from PIL import Image
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=False)

# Define the model
class ConfidenceModel(nn.Module):
    def __init__(self, input_size=8):
        super(ConfidenceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load trained model
try:
    model = ConfidenceModel()
    model.load_state_dict(torch.load("confidence_model.pth", map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit()

# Feature extraction functions
def extract_eye_angle(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_inner = face_landmarks.landmark[133]
                left_eye_outer = face_landmarks.landmark[33]
                right_eye_inner = face_landmarks.landmark[362]
                right_eye_outer = face_landmarks.landmark[263]
                left_angle = np.arctan2(left_eye_outer.y - left_eye_inner.y, 
                                       left_eye_outer.x - left_eye_inner.x) * 180 / np.pi
                right_angle = np.arctan2(right_eye_outer.y - right_eye_inner.y, 
                                        right_eye_outer.x - right_eye_inner.x) * 180 / np.pi
                return (left_angle + right_angle) / 2
        return None
    except Exception as e:
        logger.error(f"Error in extract_eye_angle: {e}")
        return None

def extract_posture_features(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            x1, y1, z1 = left_shoulder.x, left_shoulder.y, left_shoulder.z
            x2, y2, z2 = right_shoulder.x, right_shoulder.y, right_shoulder.z
            shoulder_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
            screen_vector = np.array([1, 0, 0])
            theta = np.arccos(np.dot(shoulder_vector, screen_vector) /
                             (np.linalg.norm(shoulder_vector) * np.linalg.norm(screen_vector)))
            theta_degrees = np.degrees(theta)
            return [x1, y1, z1, x2, y2, z2, theta_degrees]
        return None
    except Exception as e:
        logger.error(f"Error in extract_posture_features: {e}")
        return None

# Process image from frontend
def process_image(image_data):
    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

# Confidence status endpoint
@app.route('/confidence_status', methods=['POST'])
def confidence_status():
    data = request.get_json()
    if not data or 'image' not in data:
        response = {"Name": "Unknown", "Time": time.strftime("%H:%M:%S"), "status": "Error"}
        print(json.dumps(response))  # Print JSON response in terminal
        return json.dumps(response), 400

    name = data.get('name', 'Unknown')
    frame = process_image(data['image'])
    if frame is None:
        response = {"Name": name, "Time": time.strftime("%H:%M:%S"), "status": "Error"}
        print(json.dumps(response))  # Print JSON response in terminal
        return json.dumps(response), 500

    eye_angle = extract_eye_angle(frame)
    posture_features = extract_posture_features(frame)

    if eye_angle is not None and posture_features is not None:
        features = np.hstack((posture_features, [eye_angle]))
        features_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            output = model(features_tensor)
            prediction = torch.argmax(output).item()
            status = "Confident" if prediction == 0 else "Unconfident"
    else:
        status = "Unknown"

    response = {
        "Name": name,
        "Time": time.strftime("%H:%M:%S"),  # Format as HH:MM:SS
        "status": status
    }
    print(json.dumps(response))  # Print JSON response in terminal
    return json.dumps(response), 200, {"Content-Type": "application/json"}

@app.route('/')
def index():
    return "Confidence Detection API is running."

if __name__ == "__main__":
    try:
        logger.info("Starting Flask API...")
        port = int(os.environ.get("PORT", 5000))  # Use Render's PORT, fallback to 5000
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)  # Debug off for production
    finally:
        logger.info("Shutting down...")