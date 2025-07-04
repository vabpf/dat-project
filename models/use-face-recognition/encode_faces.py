from PIL import Image
import face_recognition
import os
import pickle
import numpy as np

# --- Configuration ---
# Path to the directory containing your known face images organized by subfolder
KNOWN_FACES_DIR = r'..\data\images'
# Path to save the encoded face data
ENCODINGS_FILE = 'encodings.pkl'
# Number of times to process each image for encoding (more can improve robustness but slower)
# Default is 1, which is usually sufficient.
NUMBER_OF_TIMES_TO_PROCESS = 1

# --- Script ---
print(f"Loading known faces from {KNOWN_FACES_DIR}...")

# Verify the directory exists before proceeding
if not os.path.exists(KNOWN_FACES_DIR):
    raise FileNotFoundError(f"The directory {KNOWN_FACES_DIR} does not exist. Please check the path.")

known_face_encodings = []
known_face_names = []
image_count = 0
face_count = 0

# Walk through the directory structure
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue # Skip if not a directory

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)

        # Skip non-image files (simple check, could be more robust)
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
             print(f"Skipping non-image file: {filename}")
             continue

        try:
            # Load the image
            pil_image = Image.open(image_path).convert('RGB')
            # Convert PIL image to numpy array (what face_recognition expects)
            image = np.array(pil_image)
            image_count += 1

            # Get face encodings from the image
            # Since images are already cropped to face, we expect exactly one face
            face_locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, face_locations, num_jitters=NUMBER_OF_TIMES_TO_PROCESS)

            if len(encodings) > 1:
                 print(f"Warning: Found more than one face in {image_path}. Using the first one.")
                 # Could add logic here to pick the largest face if needed, but for cropped
                 # faces, this warning indicates a data issue.
            elif len(encodings) == 0:
                 print(f"Warning: No face found in {image_path}. Skipping.")
                 continue # Skip this image

            # Add the encoding and name to our lists
            # We store the name multiple times, once for each image encoding
            known_face_encodings.extend(encodings) # encodings is a list, could have 0 or 1 element here
            known_face_names.extend([name] * len(encodings)) # Add the name for each encoding found

            face_count += len(encodings)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue # Continue with the next image

print(f"\nFinished processing.")
print(f"Total images found: {image_count}")
print(f"Total face encodings generated: {face_count}")
print(f"Total unique individuals processed: {len(os.listdir(KNOWN_FACES_DIR))}") # This assumes only person folders are direct children

# Check if any encodings were generated
if not known_face_encodings:
    print("No face encodings were generated. Please check your KNOWN_FACES_DIR path and image files.")
else:
    # Save the face encodings and names to a file
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

    print(f"Face encodings and names saved to {ENCODINGS_FILE}")