import cv2
import numpy as np
from flask import Flask, render_template_string, Response, request, redirect, url_for, session
import threading
import time

try:
    import RPi.GPIO as GPIO
    RPI_ENV = True
except ImportError:
    RPI_ENV = False

from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "raspi_smart_sprayer_secret"

# ---------------- CONFIGURATION ---------------- #
PUMP_GPIO_PIN = 18
PUMP_ACTIVE_HIGH = True
PUMP_DEFAULT_DURATION = 5

# Plant-specific models mapping
PLANT_MODELS = {
    "wheat": {
        "path": "models/wheat_model.h5",
        "classes": ["Healthy", "Rust", "Blight"]
    },
    "rice": {
        "path": "models/rice_model.h5",
        "classes": ["Healthy", "Brown Spot", "Leaf Blast"]
    },
    "cotton": {
        "path": "models/cotton_model.h5",
        "classes": ["Healthy", "Wilt", "Leaf Curl"]
    }
}

# ------------------------------------------------ #

# GPIO Setup
if RPI_ENV:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_GPIO_PIN, GPIO.OUT)
    GPIO.output(PUMP_GPIO_PIN, not PUMP_ACTIVE_HIGH)

# Camera Setup
camera = cv2.VideoCapture(0)

# Global state
current_frame = None
analysis_result = None
models_cache = {}

# HTML Templates
select_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Select Plant</title>
</head>
<body style="font-family: Arial; text-align:center; margin-top:50px;">
    <h1>Select Plant Type</h1>
    <form method="post" action="/select">
        <select name="plant" style="padding:10px; font-size:16px;">
            {% for plant in plants %}
            <option value="{{ plant }}">{{ plant.capitalize() }}</option>
            {% endfor %}
        </select>
        <br><br>
        <button type="submit" style="padding:10px 20px; font-size:18px;">Proceed</button>
    </form>
</body>
</html>
"""

main_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Sprayer</title>
    <script>
        function checkPlant() {
            fetch('/check').then(r => r.json()).then(data => {
                document.getElementById('result').innerText = data.result;
                if (data.severity != 'None') {
                    document.getElementById('sprayBtn').disabled = false;
                }
            });
        }
        function spray() {
            fetch('/spray').then(r => r.json()).then(data => {
                alert(data.message);
            });
        }
    </script>
</head>
<body style="font-family: Arial; text-align:center;">
    <h1>Smart Pesticide Sprayer - {{ plant.capitalize() }}</h1>
    <div>
        <img src="/video_feed" width="640" height="480"/>
    </div>
    <br>
    <button onclick="checkPlant()" style="padding:10px 20px; font-size:18px;">Check Plant</button>
    <p id="result">No analysis yet.</p>
    <button id="sprayBtn" onclick="spray()" style="padding:10px 20px; font-size:18px; background-color:red; color:white;" disabled>SPRAY</button>
</body>
</html>
"""

# Helpers
def load_plant_model(plant):
    if plant not in PLANT_MODELS:
        return None, []
    if plant not in models_cache:
        try:
            models_cache[plant] = load_model(PLANT_MODELS[plant]["path"])
        except:
            models_cache[plant] = None
    return models_cache[plant], PLANT_MODELS[plant]["classes"]

def analyze_frame(frame, plant):
    model, classes = load_plant_model(plant)
    if model:
        img = cv2.resize(frame, (128,128))
        img = img.astype("float32")/255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        confidence = float(preds[idx])
        label = classes[idx]
        severity = "None"
        if label != "Healthy":
            if confidence > 0.8:
                severity = "High"
            elif confidence > 0.5:
                severity = "Medium"
            else:
                severity = "Low"
        return f"{label} ({confidence:.2f})", severity
    else:
        return "Mock: Possible infection detected", "Medium"

def trigger_pump():
    if RPI_ENV:
        GPIO.output(PUMP_GPIO_PIN, PUMP_ACTIVE_HIGH)
        time.sleep(PUMP_DEFAULT_DURATION)
        GPIO.output(PUMP_GPIO_PIN, not PUMP_ACTIVE_HIGH)
    else:
        print("[SIM] Pump activated for", PUMP_DEFAULT_DURATION, "seconds")

# Routes
@app.route("/")
def index():
    return render_template_string(select_template, plants=PLANT_MODELS.keys())

@app.route("/select", methods=["POST"])
def select():
    plant = request.form.get("plant")
    session["plant"] = plant
    return redirect(url_for("main"))

@app.route("/main")
def main():
    plant = session.get("plant", None)
    if not plant:
        return redirect(url_for("index"))
    return render_template_string(main_template, plant=plant)

@app.route("/video_feed")
def video_feed():
    def gen():
        global current_frame
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            current_frame = frame
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/check")
def check():
    global analysis_result
    plant = session.get("plant", None)
    if not plant or current_frame is None:
        return {"result": "No frame available", "severity": "None"}
    result, severity = analyze_frame(current_frame, plant)
    analysis_result = (result, severity)
    return {"result": result, "severity": severity}

@app.route("/spray")
def spray():
    if not analysis_result:
        return {"message": "No analysis result yet"}
    _, severity = analysis_result
    if severity in ["Medium", "High"]:
        threading.Thread(target=trigger_pump).start()
        return {"message": "Pump activated!"}
    else:
        return {"message": "Spray not needed."}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)