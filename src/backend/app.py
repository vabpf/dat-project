import os
import numpy as np
import json
import base64
import cv2 # For image decoding
import re # To handle base64 prefix

import tensorflow as tf
from tensorflow.keras.models import Model

# from deepface import DeepFace # Not directly used, can be removed if not needed elsewhere
# from tensorflow.keras.models import Model # Duplicate import
from sklearn.metrics.pairwise import cosine_similarity
import logging
from flask import Flask, render_template, request, jsonify
from tqdm import tqdm

from load_custom_arcface_model import load_custom_arcface_model
from represent_with_custom_model import represent_with_custom_model

# Suppress TensorFlow logging messages
tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO) # Enable basic logging for our app

app = Flask(__name__, template_folder='/home/user/dat-project/src/frontend')

# --- Configuration ---
CUSTOM_ARCFACE_WEIGHTS_PATH = "/home/user/dat-project/models/weights/finetuned_arcface_2.h5"
KNOWN_EMBEDDINGS_FILE = "/home/user/dat-project/src/backend/mean_embeddings.json"

MODEL_NAME= "ArcFace"
DETECTOR_BACKEND = 'mtcnn'
ALIGN = True
ENFORCE_FACE_DETECTION = True
SIMILARITY_THRESHOLD = 0.50 # ## FIX: Define a threshold for a "match". Cosine similarity for ArcFace is often > 0.5 for a match.

# Global variables
custom_arcface_embedding_model = None
known_faces_data = {} # ## FIX: Load as a dictionary for easier lookup

# --- Helper Function to Load Known Embeddings ---
def load_known_embeddings(embeddings_file: str) -> dict: # ## FIX: Return a dict
    if not os.path.exists(embeddings_file):
        logging.error(f"FATAL: Known embeddings file not found at {embeddings_file}.")
        return {}
    with open(embeddings_file, 'r') as f:
        data = json.load(f)
        logging.info(f"Loaded mean embeddings for {len(data)} persons.")
    return data

# --- Initialize Model and Known Embeddings ---
with app.app_context():
    try:
        custom_arcface_embedding_model = load_custom_arcface_model(CUSTOM_ARCFACE_WEIGHTS_PATH)
        known_faces_data = load_known_embeddings(KNOWN_EMBEDDINGS_FILE)
    except Exception as e:
        logging.error(f"ERROR during startup: Could not load models or known embeddings. {e}", exc_info=True)
        # In a real app, you might want this to prevent startup
        # import sys
        # sys.exit(1)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def find_most_similar_face(target_embedding: np.ndarray, known_faces_data: dict) -> tuple:
    """
    Finds the known face with the highest cosine similarity to the target embedding.
    Returns (match_info, similarity) or (None, -1) if no match found.
    """
    if not known_faces_data:
        return None, -1

    max_similarity = -1.0
    most_similar_person_name = None

    # ## FIX: Iterate through the dictionary's items (name, embedding)
    for person_name, known_embedding in tqdm(known_faces_data.items(), desc="Comparing Embeddings", leave=False, ncols=80):
        # Ensure embeddings are in the correct shape for cosine_similarity
        sim = cosine_similarity([target_embedding], [known_embedding])[0][0]

        if sim > max_similarity:
            max_similarity = sim
            most_similar_person_name = person_name

    # ## FIX: Check if the best similarity is above our defined threshold
    if max_similarity >= SIMILARITY_THRESHOLD:
        match_info = {"name": most_similar_person_name}
        return match_info, max_similarity
    else:
        return None, max_similarity # Return the highest similarity even if it's not a match

@app.route('/compare_face', methods=['POST'])
def compare_face():
    """
    Receives an image, finds its embedding, and compares it to known faces.
    Returns a consistent JSON response for all cases.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "Invalid JSON payload or 'image' key missing"}), 400

        # ## FIX: Correctly handle the base64 string from the frontend
        base64_image_data = data['image']
        # The frontend sends just the data, but robustly handle if the prefix is there
        if ',' in base64_image_data:
            base64_image_data = base64_image_data.split(',')[1]

        img_bytes = base64.b64decode(base64_image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"status": "error", "message": "Could not decode base64 image"}), 400

        embedding_objs = represent_with_custom_model(
            img_path=img_bgr,
            custom_model=custom_arcface_embedding_model,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False, # ## FIX: Set to False to handle "no face" case gracefully
            align=ALIGN,
            normalization='base'
        )

        # ## FIX: Unified and clear response logic
        if not embedding_objs:
            # Case 1: No face was detected in the image
            response_data = {
                "status": "success",
                "message": "No face detected in the frame.",
                "match": None,
                "face_coordinates": None
            }
            return jsonify(response_data), 200

        # Case 2: Face was detected, now process it
        # We'll process the first face found
        target_embedding = embedding_objs[0]['embedding']
        face_coordinates = embedding_objs[0]['facial_area']

        most_similar_match, similarity = find_most_similar_face(np.array(target_embedding), known_faces_data)

        if most_similar_match:
            # Case 2a: A match was found above the threshold
            response_data = {
                "status": "success",
                "message": "Match found.",
                "match": {
                    "name": most_similar_match["name"],
                    "similarity": float(similarity) # Ensure it's JSON serializable
                },
                "face_coordinates": face_coordinates
            }
        else:
            # Case 2b: A face was detected, but it didn't match anyone
            response_data = {
                "status": "success",
                "message": f"No match found. Best similarity: {similarity:.2f}",
                "match": None,
                "face_coordinates": face_coordinates
            }
        return jsonify(response_data), 200

    except base64.binascii.Error as e:
        logging.error(f"Base64 decoding error: {e}")
        return jsonify({"status": "error", "message": f"Invalid Base64 data received."}), 400
    except Exception as e:
        logging.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)