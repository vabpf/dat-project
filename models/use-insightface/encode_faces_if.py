import insightface
import insightface.app as app
import os
import pickle
import numpy as np
import cv2
from scipy.spatial.distance import cosine # Cosine distance is common for InsightFace embeddings

# --- Configuration ---
# Path to the directory containing your known face images organized by subfolder
KNOWN_FACES_DIR = 'data/images' # <--- **CHANGE THIS PATH**
# Path to save the encoded face data
ENCODINGS_FILE = 'encodings_if.pkl' # Using a different name to distinguish
# InsightFace model bundle name (e.g., 'buffalo_l', 'buffalo_m', 'buffalo_s')
# 'buffalo_l' is a good general-purpose model
INSIGHTFACE_MODEL_NAME = 'buffalo_l'
# Context ID for model initialization (-1 for CPU, 0 for GPU 0, etc.)
# Use -1 if you don't have a GPU or prefer CPU for encoding
CTX_ID = -1

# --- Script ---
print(f"Loading InsightFace model '{INSIGHTFACE_MODEL_NAME}'...")
# Initialize the InsightFace app/model bundle
try:
    # The app initializes detection and recognition models within the bundle
    face_rec_app = app.FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, root='~/.insightface')
    # Prepare the model for inference. det_size specifies detection resolution.
    face_rec_app.prepare(ctx_id=CTX_ID, det_size=(640, 640)) # det_size can impact performance vs detection capability
    print("InsightFace model loaded and prepared.")
except Exception as e:
    print(f"Error loading or preparing InsightFace model: {e}")
    print("Ensure you have internet access for the first run to download models.")
    print("Also check your ONNX Runtime installation and GPU availability if using CTX_ID >= 0.")
    exit()


print(f"Loading known faces from {KNOWN_FACES_DIR}...")

known_face_encodings = []
known_face_names = []
image_count = 0
face_count = 0

# Walk through the directory structure
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue # Skip if not a directory

    print(f"Processing faces for: {name}")
    person_encodings = []

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)

        # Skip non-image files
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
             # print(f"Skipping non-image file: {filename}")
             continue

        try:
            # Load the image using OpenCV (loads as BGR numpy array)
            img = cv2.imread(image_path)
            if img is None:
                 print(f"Error loading image: {image_path}. Skipping.")
                 continue

            # Use the InsightFace app to get faces (includes detection and embedding)
            # The get method returns a list of Face objects
            faces = face_rec_app.get(img)

            if len(faces) == 0:
                 print(f"Warning: No face found in {image_path}. Skipping.")
                 continue
            elif len(faces) > 1:
                 # For already cropped images, this might indicate a data issue.
                 # We'll just take the first face found.
                 print(f"Warning: Found {len(faces)} faces in {image_path}. Using the first one.")

            # Get the embedding from the first detected face
            # InsightFace Face objects contain the embedding directly
            encoding = faces[0].embedding

            person_encodings.append(encoding)
            face_count += 1

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue # Continue with the next image

    # For each person, store all their individual image embeddings and their name
    known_face_encodings.extend(person_encodings)
    known_face_names.extend([name] * len(person_encodings))


print(f"\nFinished processing.")
print(f"Total images found: {image_count}")
print(f"Total face encodings generated: {face_count}")
# This count is rough, as it assumes only person folders are direct children
try:
    unique_individuals_count = len([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
    print(f"Total unique individuals processed: {unique_individuals_count}")
except Exception:
     print("Could not count unique individuals.")


# Check if any encodings were generated
if not known_face_encodings:
    print("No face encodings were generated. Please check your KNOWN_FACES_DIR path, image files, and model loading.")
else:
    # Save the face encodings and names to a file
    # Store as a dictionary for easy loading
    # Convert list of numpy arrays to a single numpy array for slightly faster loading later
    data = {"encodings": np.array(known_face_encodings), "names": known_face_names}
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

    print(f"Face encodings and names saved to {ENCODINGS_FILE}")