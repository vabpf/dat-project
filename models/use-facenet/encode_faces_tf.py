import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import cv2
import io
from contextlib import redirect_stdout

# Import the model loading and preprocessing functions
from facenet_model import (
    load_facenet_model, preprocess_image_for_facenet, 
    load_face_detector, detect_faces, extract_face, save_debug_image
)

# --- Configuration ---
# Path to the directory containing your known face images organized by subfolder
KNOWN_FACES_DIR = '../data/images'
# Path to save the encoded face data
ENCODINGS_FILE = 'encodings_tf.pkl'
# Directory for debug images
DEBUG_DIR = 'debug_images'
# Enable debug mode
DEBUG = True

# Suppress TensorFlow logging 
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages

# Create debug directory if needed
if DEBUG and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
    print(f"Created debug directory: {DEBUG_DIR}")

# --- Script ---
print(f"Loading known faces from {KNOWN_FACES_DIR}...")

# Load the FaceNet model
model = load_facenet_model()

if model is None:
    print("Failed to load the FaceNet model. Exiting.")
    exit()

# Load the face detector - try DNN first, fallback to Haar
print("Loading face detector...")
detector = load_face_detector("dnn")
if detector is None:
    print("Failed to load face detector. Exiting.")
    exit()

# First, count the total number of images to process for the progress bar
total_images = 0
all_image_paths = []
all_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue  # Skip if not a directory
    
    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            continue
        total_images += 1
        all_image_paths.append(image_path)
        all_names.append(name)

# Process all images with a single progress bar
print(f"Processing {total_images} images across {len(os.listdir(KNOWN_FACES_DIR))} individuals...")

known_face_encodings = []
known_face_names = []
image_count = 0
face_count = 0
debug_count = 0

# Create a progress bar for all images
with tqdm(total=total_images, desc="Encoding faces") as pbar:
    for image_path, name in zip(all_image_paths, all_names):
        try:
            # Load the image - try PIL first
            try:
                image_pil = Image.open(image_path)
                # Convert PIL image to numpy array
                image = np.array(image_pil)
            except Exception as e:
                print(f"PIL failed to load {image_path}, trying OpenCV: {e}")
                # Fallback to OpenCV
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None or image.size == 0:
                print(f"Could not load image: {image_path}")
                pbar.update(1)
                continue

            image_count += 1
            
            # Detect faces in the image
            faces = detect_faces(image, detector, min_confidence=0.5)
            
            # If debug mode is enabled, save an image with face boxes
            if DEBUG:
                debug_path = os.path.join(DEBUG_DIR, f"debug_{debug_count:03d}_{os.path.basename(image_path)}")
                save_debug_image(image, faces, debug_path)
                debug_count += 1
            
            # If no faces are detected, skip the image
            if len(faces) == 0:
                print(f"No faces detected in {image_path}")
                pbar.update(1)
                continue
                
            # Process each detected face
            for face_idx, face_box in enumerate(faces):
                try:
                    # Extract the face with a margin
                    face_img = extract_face(image, face_box)
                    
                    # Skip if face extraction failed
                    if face_img is None:
                        # print(f"Face extraction failed for face {face_idx} in {image_path}")
                        continue
                    
                    # Add debugging information about face dimensions - comment out for cleaner output
                    # print(f"Face {face_idx} dimensions: {face_img.shape}")
                    
                    # Preprocess the face for FaceNet
                    processed_face = preprocess_image_for_facenet(face_img)
                    
                    # Save debug image of the extracted face
                    if DEBUG:
                        face_debug_path = os.path.join(DEBUG_DIR, f"face_{debug_count:03d}_{face_idx}_{os.path.basename(image_path)}")
                        cv2.imwrite(face_debug_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    
                    # Get the face embedding - Redirect stdout to suppress the progress bar
                    
                    
                    # Temporarily redirect stdout to suppress progress bar
                    with redirect_stdout(io.StringIO()):
                        embedding = model.embeddings(processed_face)[0]
                    
                    # Save the embedding and name
                    known_face_encodings.append(embedding)
                    known_face_names.append(name)
                    face_count += 1
                    
                    # Comment out success message for cleaner output
                    # print(f"Successfully generated embedding for face {face_idx} in {image_path}")

                except Exception as e:
                    print(f"Error processing face {face_idx} from {image_path}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            pass
        
        # Update the progress bar
        pbar.update(1)

print(f"\nFinished processing.")
print(f"Total images found: {image_count}")
print(f"Total face encodings generated: {face_count}")
print(f"Total unique individuals processed: {len(set(known_face_names))}")

# Check if any encodings were generated
if not known_face_encodings:
    print("No face encodings were generated. Please check your KNOWN_FACES_DIR path, image files, and model loading.")
else:
    # Save the face encodings and names to a file
    # Store as a dictionary for easy loading
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

    print(f"Face encodings and names saved to {ENCODINGS_FILE}")