# EdgeAI_WildFire_Detection
This Python code integrates YOLOv8n for object detection, IoT sensor fusion (temperature, gas sensors), and temporal smoothing on Jetson Nano. It processes real-time video, confirms detections using sensor thresholds, and applies a buffer to reduce false positives. The code also generates real-time alerts using MQTT for IoT notifications and Twilio for SMS alerts whenever a wildfire is confirmed

'''import cv2
import torch
import numpy as np
import time
import Adafruit_DHT
import serial
import paho.mqtt.client as mqtt
from twilio.rest import Client
from collections import deque
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Define sensor threshold values
TEMP_THRESHOLD = 45  # Celsius
GAS_THRESHOLD = 400  # PPM

# Initialize sensors
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Temporal smoothing buffer
FRAME_WINDOW = 5
detection_buffer = deque(maxlen=FRAME_WINDOW)

# MQTT Configuration
MQTT_BROKER = "mqtt.example.com"
MQTT_TOPIC = "wildfire/alert"
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

# Twilio SMS Configuration
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
RECIPIENT_PHONE_NUMBER = "recipient_phone_number"
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert():
    message = "🔥 Wildfire Detected! Immediate Action Required."
    client.publish(MQTT_TOPIC, message)
    twilio_client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print("Alert Sent: MQTT + SMS")

# MQTT Subscriber

def on_message(client, userdata, msg):
    print(f"MQTT Alert Received: {msg.payload.decode()}")

subscriber_client = mqtt.Client()
subscriber_client.connect(MQTT_BROKER, 1883, 60)
subscriber_client.subscribe(MQTT_TOPIC)
subscriber_client.on_message = on_message
subscriber_client.loop_start()

# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Object Detection
    results = model(frame)
    fire_detected = any(detection.names[0] == 'fire' for detection in results)
    
    # Read sensor data
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    gas_level = ser.readline().strip()
    
    try:
        gas_level = float(gas_level)
    except ValueError:
        gas_level = 0  # Default if reading fails
    
    # Fusion logic: Fire detected in image + sensor anomalies
    sensor_alert = (temperature and temperature > TEMP_THRESHOLD) or (gas_level > GAS_THRESHOLD)
    final_detection = fire_detected and sensor_alert
    
    # Temporal Smoothing
    detection_buffer.append(final_detection)
    confirmed_fire = sum(detection_buffer) > (FRAME_WINDOW // 2)
    
    # Trigger alert if fire is confirmed
    if confirmed_fire:
        send_alert()
    
    # Display result
    label = "FIRE DETECTED!" if confirmed_fire else "No Fire"
    color = (0, 0, 255) if confirmed_fire else (0, 255, 0)
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Wildfire Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

