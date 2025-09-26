"""
Raspberry Pi Smart Sprayer - single-file Flask app
Features:
- MJPEG live stream from Pi camera / USB webcam
- "Check Plant" button: captures current frame and runs disease detection
  - If a TensorFlow Keras model ('model.h5') is placed beside this file it will be used.
  - Otherwise, a lightweight placeholder detector (color/contour-based) runs as fallback.
- Shows disease name + severity (None / Low / Medium / High) on web UI
- Enables a red "SPRAY" button when detection indicates infection
- Clicking SPRAY activates a GPIO pin for the pump for a configurable duration

Notes:
- Tested for Raspberry Pi OS with a USB webcam or Raspberry Pi Camera (v2). For Pi Camera v2 you may need to use
  'cv2.VideoCapture(0)' or 'cv2.VideoCapture(0, cv2.CAP_V4L2)'.
- Hardware: pump controlled via a relay module or motor driver. DO NOT drive a pump directly from GPIO.
  Use a relay or MOSFET + external power supply. Use proper optoisolation where needed.

Save this file, install dependencies, then run with `python3 raspi_smart_sprayer.py`.
Open http://<raspi-ip>:5000 in your phone/laptop on the same Wi-Fi network.

"""

from flask import Flask, render_template_string, Response, jsonify, request
import threading
import time
import io
import os
import sys

# Camera and image processing
import cv2
import numpy as np

# If tensorflow is installed and model.h5 exists, we will load & use it
USE_TF = False
try:
    from tensorflow.keras.models import load_model
    USE_TF = True
except Exception as e:
    # TensorFlow may be heavy; fallback will work without it
    USE_TF = False

# GPIO setup (Raspberry Pi)
USE_GPIO = False
try:
    import RPi.GPIO as GPIO
    USE_GPIO = True
except Exception:
    USE_GPIO = False

# ---- Configuration ----
PUMP_GPIO_PIN = 18      # GPIO pin to activate relay for pump (BCM numbering)
PUMP_ACTIVE_HIGH = True # True if setting pin HIGH turns pump on via relay; else False
PUMP_DEFAULT_DURATION = 5.0  # seconds

CAMERA_INDEX = 0        # default camera index
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MODEL_PATH = 'model.h5'  # optional keras model

# ---- Flask app & Camera thread ----
app = Flask(__name__)

camera = None
frame_lock = threading.Lock()
latest_frame = None
running = True

class CameraThread(threading.Thread):
    def __init__(self, index=0):
        super().__init__()
        self.index = index
        self.cap = None
        self.daemon = True

    def run(self):
        global latest_frame, running
        # Try to open camera
        self.cap = cv2.VideoCapture(self.index)
        # Try setting resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            print('Error: Camera not opened. Check camera index or connections.', file=sys.stderr)
            return

        while running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # encode/frame
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.02)

    def get_frame(self):
        with frame_lock:
            if latest_frame is None:
                return None
            return latest_frame.copy()

# ---- Load optional ML model ----
keras_model = None
if USE_TF and os.path.exists(MODEL_PATH):
    try:
        print('Loading Keras model from', MODEL_PATH)
        keras_model = load_model(MODEL_PATH)
        print('Model loaded.')
    except Exception as e:
        print('Failed to load model:', e, file=sys.stderr)
        keras_model = None

# ---- GPIO init ----
if USE_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_GPIO_PIN, GPIO.OUT)
    # Set pump default off
    if PUMP_ACTIVE_HIGH:
        GPIO.output(PUMP_GPIO_PIN, GPIO.LOW)
    else:
        GPIO.output(PUMP_GPIO_PIN, GPIO.HIGH)
else:
    print('RPi.GPIO not available. Pump actions will be simulated.')

# ---- Helper: encode frame as JPEG for MJPEG streaming ----
def gen_mjpeg(camera_thread):
    while True:
        frame = camera_thread.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        chunk = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')

# ---- Simple placeholder detector ----
# Returns (label, confidence, severity_level)
# severity_level: 0=None, 1=Low, 2=Medium, 3=High

def placeholder_detector(frame):
    # Convert to HSV and check brown/yellow areas (common for leaf disease)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # brownish/yellowish range - approximate
    lower = np.array([5, 50, 50])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cover = (np.count_nonzero(mask) / (frame.shape[0]*frame.shape[1]))
    # Map cover fraction to severity
    if cover < 0.003:
        return ('Healthy', 1.0 - cover, 0)
    elif cover < 0.02:
        return ('Possible Leaf Spot', min(0.9, cover*50), 1)
    elif cover < 0.08:
        return ('Infected: Leaf Spot', min(0.95, cover*10), 2)
    else:
        return ('Severe Infection', min(0.99, cover*5), 3)

# ---- Optional Keras inference wrapper ----
# This expects the Keras model to accept an image array and return a vector of probabilities
# Provide mapping from class idx -> name in CLASS_NAMES if using a model.
CLASS_NAMES = ['Healthy', 'Disease_A', 'Disease_B', 'Disease_C']

def keras_detector(frame):
    global keras_model
    # Preprocess - resize to model input
    h, w = frame.shape[:2]
    input_size = (224, 224)
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = keras_model.predict(arr)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = preds
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f'Class {idx}'
    confidence = float(probs[idx])
    # Map confidence to severity heuristically
    if label == 'Healthy':
        severity = 0 if confidence > 0.6 else 1
    else:
        if confidence < 0.5:
            severity = 1
        elif confidence < 0.8:
            severity = 2
        else:
            severity = 3
    return (label, confidence, severity)

# ---- Pump control ----
pump_lock = threading.Lock()

def activate_pump(duration=PUMP_DEFAULT_DURATION):
    # Activate pump for <duration> seconds
    with pump_lock:
        if USE_GPIO:
            try:
                if PUMP_ACTIVE_HIGH:
                    GPIO.output(PUMP_GPIO_PIN, GPIO.HIGH)
                else:
                    GPIO.output(PUMP_GPIO_PIN, GPIO.LOW)
                time.sleep(duration)
            finally:
                if PUMP_ACTIVE_HIGH:
                    GPIO.output(PUMP_GPIO_PIN, GPIO.LOW)
                else:
                    GPIO.output(PUMP_GPIO_PIN, GPIO.HIGH)
                return True
        else:
            # Simulate
            print(f"[SIMULATION] Activating pump for {duration} seconds")
            time.sleep(duration)
            print('[SIMULATION] Pump deactivated')
            return True

# ---- Flask routes ----
INDEX_HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RasPi Smart Sprayer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, Helvetica, sans-serif; padding: 12px; background:#f4f6f8; }
    .container { max-width:900px; margin:0 auto; }
    h1 { text-align:center; }
    .video { text-align:center; margin-bottom:12px; }
    img#live { border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,0.12); max-width:100%; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
    button { padding:12px 18px; font-size:16px; border-radius:8px; border: none; cursor:pointer; }
    button.primary { background:#2b7cff; color:white; }
    button.danger { background:#e63737; color:white; }
    button.disabled { background:#bbb; color:#444; cursor:not-allowed; }
    .result { margin-top:16px; text-align:center; background:white; padding:12px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
    .severity { font-weight:700; padding:6px 12px; border-radius:6px; display:inline-block; }
    .sev-0 { background:#cdeacb; color:#0a6b2f; }
    .sev-1 { background:#fff3cd; color:#7a5f00; }
    .sev-2 { background:#ffd9d0; color:#8b2a1a; }
    .sev-3 { background:#f8d7da; color:#721c24; }
    .small { font-size:14px; color:#444; }
  </style>
</head>
<body>
  <div class="container">
    <h1>RasPi Smart Sprayer</h1>
    <div class="video">
      <img id="live" src="/video_feed" alt="Live camera stream">
    </div>
    <div class="controls">
      <button id="checkBtn" class="primary">Check Plant</button>
      <button id="sprayBtn" class="danger disabled" disabled>SPRAY</button>
    </div>
    <div id="result" class="result" style="display:none;">
      <div id="label" style="font-size:18px; font-weight:700;">Label</div>
      <div id="conf" class="small">Confidence: --</div>
      <div id="sev" style="margin-top:8px;"></div>
      <div id="log" class="small" style="margin-top:8px;color:#666"></div>
    </div>
  </div>

<script>
const checkBtn = document.getElementById('checkBtn');
const sprayBtn = document.getElementById('sprayBtn');
const resultBox = document.getElementById('result');
const labelEl = document.getElementById('label');
const confEl = document.getElementById('conf');
const sevEl = document.getElementById('sev');
const logEl = document.getElementById('log');

checkBtn.onclick = async () => {
  checkBtn.disabled = true;
  checkBtn.innerText = 'Checking...';
  logEl.innerText = 'Capturing frame and analyzing...';
  try {
    const r = await fetch('/detect', {method:'POST'});
    const j = await r.json();
    labelEl.innerText = j.label;
    confEl.innerText = 'Confidence: ' + (j.confidence*100).toFixed(1) + '%';
    let sevText = '';
    let sevClass = '';
    if (j.severity === 0) { sevText = 'None'; sevClass='sev-0'; sprayBtn.classList.add('disabled'); sprayBtn.disabled=true; }
    if (j.severity === 1) { sevText = 'Low'; sevClass='sev-1'; sprayBtn.classList.add('disabled'); sprayBtn.disabled=true; }
    if (j.severity === 2) { sevText = 'Medium'; sevClass='sev-2'; sprayBtn.classList.remove('disabled'); sprayBtn.disabled=false; }
    if (j.severity === 3) { sevText = 'High'; sevClass='sev-3'; sprayBtn.classList.remove('disabled'); sprayBtn.disabled=false; }
    sevEl.innerHTML = '<span class="severity '+sevClass+'">Severity: '+sevText+'</span>';
    resultBox.style.display = 'block';
    logEl.innerText = j.log || '';
  } catch (e) {
    logEl.innerText = 'Error: ' + e.toString();
  }
  checkBtn.disabled = false;
  checkBtn.innerText = 'Check Plant';
}

sprayBtn.onclick = async () => {
  if (sprayBtn.disabled) return;
  const confirmSpray = confirm('Activate pump for spraying?');
  if (!confirmSpray) return;
  sprayBtn.disabled = true;
  sprayBtn.innerText = 'Spraying...';
  try {
    const r = await fetch('/spray', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({duration: 5})});
    const j = await r.json();
    alert('Spray result: ' + j.status);
    logEl.innerText = 'Last spray: ' + j.status + ' (' + j.duration + 's)';
  } catch (e) {
    alert('Spray failed: ' + e);
  }
  sprayBtn.disabled = false;
  sprayBtn.innerText = 'SPRAY';
}
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_mjpeg(cam_thread), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    # Capture latest frame and run detection
    frame = cam_thread.get_frame()
    if frame is None:
        return jsonify({'error':'no_frame', 'message':'No camera frame available.'}), 503

    # Optionally run keras model
    if keras_model is not None:
        try:
            label, conf, severity = keras_detector(frame)
            return jsonify({'label': label, 'confidence': conf, 'severity': severity, 'log':'Used Keras model.'})
        except Exception as e:
            print('Keras detection failed:', e, file=sys.stderr)
            # fallback to placeholder

    # Placeholder detection
    label, conf, severity = placeholder_detector(frame)
    return jsonify({'label': label, 'confidence': conf, 'severity': severity, 'log':'Used placeholder detector.'})

@app.route('/spray', methods=['POST'])
def spray():
    body = request.get_json() or {}
    duration = float(body.get('duration', PUMP_DEFAULT_DURATION))
    # Activate pump in a separate thread so HTTP can return quickly
    t = threading.Thread(target=activate_pump, args=(duration,))
    t.start()
    return jsonify({'status':'activated', 'duration': duration})

# ---- Start camera thread and app ----
if __name__ == '__main__':
    cam_thread = CameraThread(CAMERA_INDEX)
    cam_thread.start()
    try:
        # Run flask on all interfaces so it's accessible on the Pi's Wi-Fi IP
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        running = False
        time.sleep(0.5)
        if USE_GPIO:
            GPIO.cleanup()
