import cv2
import numpy as np
import dlib
import threading
import pygame
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sqlite3
import datetime

# Initialize the Dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize or connect to the SQLite database
conn = sqlite3.connect('drowsiness_log.db')
cursor = conn.cursor()

# Create a table to store drowsiness entries
cursor.execute('''
    CREATE TABLE IF NOT EXISTS drowsiness (
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP
    )
''')


# Constants for drowsiness detection
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.5
CONSEC_FRAMES = 18

# Constants for vehicle simulation
vehicle_speed = 60
max_speed = 100
min_speed = 0
speed_increment = 5
stop_speed = 0
hazard_lights_on = False

# Initialize the Pygame mixer for the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm1.mp3")

# Variables to track alarm state and timing
alarm_on = False
alarm_start_time = 0
ALARM_DURATION = 3

# Colors for the box around the face
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Define a simple CNN model with 4 output classes
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes: closed, open, no_yawn, yawn

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load the pre-trained model weights
model = DrowsinessCNN()
model.load_state_dict(torch.load('drowsiness_model_1.pth'))

# Set the model to evaluation mode
model.eval()

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    if len(eye) != 6:
        raise ValueError("Eye must have exactly six points")
    # Calculate the horizontal distance between the left and right eye landmarks
    left_eye_width = euclidean_distance(eye[0], eye[3])
    # Calculate the vertical distance between the top and bottom eye landmarks
    left_eye_height = (euclidean_distance(eye[1], eye[5]) + euclidean_distance(eye[2], eye[4]))
    # Avoid division by zero
    if left_eye_height == 0:
        return 0.0
    # Calculate EAR
    ear = left_eye_height / (2.0 * left_eye_width)
    return ear

# Function to detect yawning based on mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    if len(mouth) != 20:
        return 0.0
    # Calculate MAR
    A = euclidean_distance(mouth[13], mouth[19])
    B = euclidean_distance(mouth[14], mouth[18])
    C = euclidean_distance(mouth[15], mouth[17])
    # Avoid division by zero
    if C == 0:
        return 0.0
    MAR = (A + B) / (2.0 * C)
    return MAR

# Function to slow down the vehicle
def slow_down():
    global vehicle_speed
    if vehicle_speed > stop_speed:
        vehicle_speed -= speed_increment
    else:
        vehicle_speed = stop_speed

# Function to speed up the vehicle
def speed_up():
    global vehicle_speed
    if vehicle_speed < max_speed:
        vehicle_speed += speed_increment

# Function to stop the vehicle
def stop_vehicle():
    global vehicle_speed
    vehicle_speed = stop_speed

# Function to toggle hazard lights
def toggle_hazard_lights():
    global hazard_lights_on
    hazard_lights_on = not hazard_lights_on

# Function to play the alarm sound
def play_alarm():
    while True:
        if alarm_on:
            alarm_sound.play()
            time.sleep(1)  # Play the alarm for 1 second
        else:
            pygame.time.delay(1000)  # Delay to avoid busy-wait

# Function to display alert text and change box color
def display_alert(frame):
    cv2.putText(frame, "Drowsiness/Yawning Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return RED  # Change the box color to red

def log_drowsiness():
    timestamp = datetime.datetime.now()

    # Insert drowsiness entry into the database
    cursor.execute('INSERT INTO drowsiness (timestamp) VALUES (?)', (timestamp,))
    conn.commit()
    print("Log inserted")

# Start the alarm thread
alarm_thread = threading.Thread(target=play_alarm)
alarm_thread.daemon = True
alarm_thread.start()

# Open the video camera
cap = cv2.VideoCapture(0)

# Main loop
frame_counter = 0
box_color = GREEN
alert_text_displayed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Dlib detector
    faces = detector(gray)
    alarm_triggered = False

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        mouth = [landmarks.part(i) for i in range(48, 68)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
            frame_counter += 1

            if frame_counter >= CONSEC_FRAMES:
                alarm_triggered = True
        else:
            frame_counter = 0

    if alarm_triggered:
        if not alarm_on:
            alarm_on = True
            alarm_start_time = time.time()
            slow_down()
            toggle_hazard_lights()
            box_color = display_alert(frame)
            alert_text_displayed = True
            log_drowsiness()
    elif alarm_on and time.time() - alarm_start_time >= ALARM_DURATION:
        alarm_on = False
        box_color = GREEN
        alert_text_displayed = False

    if alert_text_displayed:
        cv2.putText(frame, "Drowsiness/Yawning Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
