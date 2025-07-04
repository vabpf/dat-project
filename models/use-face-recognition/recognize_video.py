import face_recognition
import cv2
import numpy as np
import pickle
import os
import time
from flask import Flask, render_template, Response, request
import threading
import queue

# --- Configuration ---
ENCODINGS_FILE = 'encodings.pkl'
# How far apart can faces be and still be considered a match? Lower is stricter.
# 0.6 is a common default. You might need to tune this based on your data/results.
FACE_RECOGNITION_TOLERANCE = 0.6
# Scale factor for resizing the frame to speed up face detection (e.g., 0.25 means process at 1/4 size)
# Lower values are faster but might miss smaller faces. Set to 1.0 for no resizing.
FRAME_PROCESS_SCALE = 0.5
# How many frames to skip between full recognition passes (to save CPU)
# Recognition is done every `RECOGNITION_FRAME_INTERVAL` frames. Face detection is done on every frame.
RECOGNITION_FRAME_INTERVAL = 5

# Initialize Flask app
app = Flask(__name__)

# Global variables
video_capture = None
known_face_encodings = []
known_face_names = []
# Shared recognition results
current_face_locations = []
current_face_names = []
# Threading and synchronization
lock = threading.Lock()
frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
result_ready = threading.Event()
processing_thread = None
running = False
should_exit = False
frame_count = 0
start_time = time.time()

def load_encodings():
    global known_face_encodings, known_face_names
    
    print(f"Loading face encodings from {ENCODINGS_FILE}...")
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
        print(f"Loaded {len(known_face_encodings)} known face encodings for {len(set(known_face_names))} unique individuals.")
        return True
    except FileNotFoundError:
        print(f"Error: {ENCODINGS_FILE} not found. Please run encode_faces.py first.")
        return False
    except Exception as e:
        print(f"Error loading encodings file: {e}")
        return False

def initialize_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0) # 0 is typically the default webcam
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False
    return True

# Function to handle Vietnamese text display
def draw_text_with_vietnamese(img, text, pos, font, font_scale, color, thickness):
    # Convert to PIL Image for better text rendering
    try:
        import PIL.Image, PIL.ImageDraw, PIL.ImageFont
        
        # Create a temporary transparent image for text
        pil_img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(pil_img)
        
        # Attempt to load a font that supports Vietnamese
        # Default to a system font if no specific font is available
        try:
            # Try to use a common font that supports Vietnamese
            font = PIL.ImageFont.truetype("arial.ttf", int(font_scale * 30))
        except:
            # Fallback to default
            font = PIL.ImageFont.load_default()
            
        # Draw the text
        draw.text(pos, text, font=font, fill=color[::-1])  # Convert BGR to RGB
        
        # Convert back to numpy array for OpenCV
        result_img = np.array(pil_img)
        return result_img
    except ImportError:
        # If PIL is not available, fall back to OpenCV's putText
        cv2.putText(img, text, pos, font, font_scale, color, thickness)
        return img

def process_frame_thread():
    """Background thread function for face recognition processing"""
    global current_face_locations, current_face_names, should_exit
    
    recognition_count = 0
    
    while not should_exit:
        try:
            # Get the latest frame from queue, non-blocking
            frame = frame_queue.get(block=False)
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_PROCESS_SCALE, fy=FRAME_PROCESS_SCALE)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Do face recognition
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 
                                                        tolerance=FACE_RECOGNITION_TOLERANCE)
                name = "Unknown"
                
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if face_distances[best_match_index] < FACE_RECOGNITION_TOLERANCE:
                        name = known_face_names[best_match_index]
                
                names.append(name)
            
            # Update the shared variables with a lock
            with lock:
                current_face_locations = face_locations
                current_face_names = names
                result_ready.set()  # Signal that new results are ready
            
            recognition_count += 1
            
        except queue.Empty:
            # No new frame to process, sleep briefly
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(0.1)
    
    print(f"Face recognition thread exited after processing {recognition_count} frames")

def apply_recognition_results(frame):
    """Apply the current recognition results to the frame"""
    # Use the latest recognition results
    with lock:
        face_locations = current_face_locations.copy()
        face_names = current_face_names.copy()
    
    # Apply the results to the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top = int(top / FRAME_PROCESS_SCALE)
        right = int(right / FRAME_PROCESS_SCALE)
        bottom = int(bottom / FRAME_PROCESS_SCALE)
        left = int(left / FRAME_PROCESS_SCALE)
        
        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), color, cv2.FILLED)
        
        # Use the function to properly display Vietnamese text
        font = cv2.FONT_HERSHEY_DUPLEX
        try:
            frame = draw_text_with_vietnamese(frame, name, (left + 6, bottom + 3), font, 0.6, (255, 255, 255), 1)
        except:
            cv2.putText(frame, name, (left + 6, bottom + 3), font, 0.6, (255, 255, 255), 1)
    
    return frame

def generate_frames():
    """Generator function for video streaming"""
    global running, video_capture, frame_count, processing_thread, should_exit
    
    if not running:
        if not initialize_camera():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + 
                   b'\r\n')
            return
        
        # Start the processing thread
        should_exit = False
        processing_thread = threading.Thread(target=process_frame_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        running = True
        frame_count = 0
    
    while True:
        # Get a new frame from camera
        success, frame = video_capture.read()
        if not success:
            print("Error reading frame from webcam.")
            break
        
        frame_count += 1
        
        # Add frame to queue for processing (non-blocking)
        try:
            # Always replace the current frame in the queue with the newest one
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip if queue is full
        
        # Apply latest recognition results to the current frame
        labeled_frame = apply_recognition_results(frame)
        
        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', labeled_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_camera():
    global running, video_capture, should_exit, processing_thread
    
    with lock:
        if running and video_capture is not None:
            should_exit = True
            if processing_thread is not None and processing_thread.is_alive():
                processing_thread.join(timeout=1.0)
            
            video_capture.release()
            running = False
            
            end_time = time.time()
            print(f"\nRecognition stopped after {frame_count} frames.")
            if frame_count > 0:
                print(f"Average FPS: {frame_count / (end_time - start_time):.2f}")
    
    return "Camera stopped"

if __name__ == '__main__':
    # Load face encodings before starting the app
    if not load_encodings():
        exit(1)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #333;
        }
        .video-container {
            margin: 20px auto;
            max-width: 800px;
        }
        img {
            width: 100%;
            border: 1px solid #ddd;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Face Recognition Live Stream</h1>
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" alt="Video Stream">
    </div>
    <div class="controls">
        <button onclick="stopCamera()">Stop Camera</button>
    </div>
    
    <script>
        function stopCamera() {
            fetch('/stop', {method: 'POST'})
                .then(response => {
                    console.log('Camera stopped');
                    alert('Camera has been stopped');
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
    </script>
</body>
</html>
        ''')
    
    # Start the Flask app
    print("Starting Flask server. Access the webcam stream at http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False)