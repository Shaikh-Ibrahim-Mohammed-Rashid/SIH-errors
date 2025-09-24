from flask import Flask, render_template, Response
import RPi.GPIO as GPIO
import cv2
import time
import socket

# ===================== GPIO SETUP =====================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Motor direction pins
Motor_In1 = 29
Motor_In2 = 31
Motor_In3 = 33
Motor_In4 = 35

# Water Pump Pin
PUMP_PIN = 37

# ENABLE pins for L298N
ENA = 32
ENB = 36

# Setup pins
GPIO.setup(Motor_In1, GPIO.OUT)
GPIO.setup(Motor_In2, GPIO.OUT)
GPIO.setup(Motor_In3, GPIO.OUT)
GPIO.setup(Motor_In4, GPIO.OUT)
GPIO.setup(PUMP_PIN, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# PWM setup
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# Initial states
GPIO.output(Motor_In1, False)
GPIO.output(Motor_In2, False)
GPIO.output(Motor_In3, False)
GPIO.output(Motor_In4, False)
GPIO.output(PUMP_PIN, False)

# Default motor speed
motor_speed = 50

def set_motor_speed(speed):
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

set_motor_speed(motor_speed)

# ===================== CAMERA =====================
video_capture = cv2.VideoCapture(0)  # Use index 0 only
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===================== NETWORK =====================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

Url_Address = get_local_ip()

# ===================== FLASK =====================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("temp.html", HTML_address=Url_Address)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== MOTOR ROUTES =====================
@app.route('/Forward')
def forward():
    set_motor_speed(motor_speed)
    GPIO.output(Motor_In1, True)
    GPIO.output(Motor_In2, False)
    GPIO.output(Motor_In3, True)
    GPIO.output(Motor_In4, False)
    return render_template("temp.html", HTML_address=Url_Address)

@app.route('/Backward')
def backward():
    set_motor_speed(motor_speed)
    GPIO.output(Motor_In1, False)
    GPIO.output(Motor_In2, True)
    GPIO.output(Motor_In3, False)
    GPIO.output(Motor_In4, True)
    return render_template("temp.html", HTML_address=Url_Address)

@app.route('/left')
def left():
    set_motor_speed(motor_speed)
    GPIO.output(Motor_In1, True)
    GPIO.output(Motor_In2, False)
    GPIO.output(Motor_In3, False)
    GPIO.output(Motor_In4, True)
    return render_template("temp.html", HTML_address=Url_Address)

@app.route('/right')
def right():
    set_motor_speed(motor_speed)
    GPIO.output(Motor_In1, False)
    GPIO.output(Motor_In2, True)
    GPIO.output(Motor_In3, True)
    GPIO.output(Motor_In4, False)
    return render_template("temp.html", HTML_address=Url_Address)

@app.route('/stop')
def stop():
    set_motor_speed(0)
    GPIO.output(Motor_In1, False)
    GPIO.output(Motor_In2, False)
    GPIO.output(Motor_In3, False)
    GPIO.output(Motor_In4, False)
    return render_template("temp.html", HTML_address=Url_Address)

# ===================== PUMP =====================
@app.route('/spray')
def spray():
    GPIO.output(PUMP_PIN, True)
    time.sleep(3)
    GPIO.output(PUMP_PIN, False)
    return render_template("temp.html", HTML_address=Url_Address)

# ===================== SPEED CONTROL =====================
@app.route('/speed/<int:speed>')
def set_speed(speed):
    global motor_speed
    motor_speed = max(0, min(100, speed))
    set_motor_speed(motor_speed)
    return f"Speed set to {motor_speed}%"

# ===================== CLEANUP =====================
def cleanup():
    pwm_a.stop()
    pwm_b.stop()
    GPIO.output(PUMP_PIN, False)
    GPIO.cleanup()
    video_capture.release()

if __name__ == "__main__":
    try:
        print(f"Starting server on http://{Url_Address}:8080")
        app.run(host="0.0.0.0", port=8080, threaded=True, debug=True)
    except KeyboardInterrupt:
        cleanup()
