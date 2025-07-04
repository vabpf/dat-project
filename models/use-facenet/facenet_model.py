# Define the InceptionResNetV1 model architecture as used in FaceNet implementations
# This is a simplified version, assuming standard layers.
# You might need to adjust layer names/parameters slightly based on the exact weights file source.
import cv2
import numpy as np
from keras_facenet import FaceNet
from PIL import Image
import os

def load_facenet_model(weights_path=None):
    """
    Loads the FaceNet model using keras-facenet package.
    Note: weights_path is kept for compatibility but not used as keras-facenet loads its own weights
    """
    print("Loading FaceNet model...")
    try:
        # Initialize the FaceNet model from keras-facenet package
        model = FaceNet()
        print("FaceNet model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading FaceNet model: {e}")
        return None

def load_face_detector(detector_type="dnn"):
    """
    Loads a face detector model.
    Args:
        detector_type: Type of detector to use ("dnn" or "haar")
    Returns:
        Face detector or a tuple containing the model and required parameters
    """
    if detector_type == "dnn":
        try:
            # Define paths to the DNN model files
            prototxt_path = os.path.join('models', 'deploy.prototxt.txt')
            caffemodel_path = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')
            
            # Check if model files exist
            if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
                print(f"DNN model files not found at {prototxt_path} or {caffemodel_path}")
                print("Falling back to Haar Cascade detector")
                return load_face_detector("haar")
            
            # Load the DNN face detector
            detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            print("Loaded DNN face detector")
            return ("dnn", detector)
        except Exception as e:
            print(f"Error loading DNN face detector: {e}")
            print("Falling back to Haar Cascade detector")
            return load_face_detector("haar")
    else:  # "haar" or fallback
        try:
            # Load OpenCV's built-in face detector (Haar Cascade)
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Loaded Haar Cascade face detector")
            return ("haar", detector)
        except Exception as e:
            print(f"Error loading Haar Cascade face detector: {e}")
            return None

def detect_faces(image_array, detector, min_confidence=0.5):
    """
    Detects faces in an image using the specified detector.
    
    Args:
        image_array: The image as a numpy array
        detector: Tuple of (detector_type, detector_model)
        min_confidence: Minimum confidence threshold for face detection (for DNN)
        
    Returns:
        List of face bounding boxes
    """
    if image_array is None:
        return []
    
    detector_type, detector_model = detector
    
    # Get image dimensions
    h, w = image_array.shape[:2]
    
    # Set default RGB flag based on number of channels
    is_rgb = len(image_array.shape) == 3 and image_array.shape[2] >= 3
    
    if detector_type == "dnn":
        # DNN expects RGB input - ensure image is in correct format
        if is_rgb:
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image_array, (300, 300)), 
                1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Set the blob as input to the network
            detector_model.setInput(blob)
            
            # Perform detection
            detections = detector_model.forward()
            
            # Process detections
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter by confidence
                if confidence > min_confidence:
                    # Get bounding box coordinates (normalized)
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    # Convert to integers
                    x1, y1, x2, y2 = box.astype('int')
                    # Convert to xywh format expected by the rest of the code
                    faces.append((x1, y1, x2-x1, y2-y1))
            
            # Comment out face detection count for cleaner output
            # print(f"DNN detector found {len(faces)} faces")
            return faces
        else:
            print("DNN detector requires RGB images. Image has wrong format.")
            return []
    else:  # "haar"
        # Convert to grayscale for Haar detection if needed
        if is_rgb:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Detect faces with more relaxed parameters
        faces = detector_model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # Lower value for more detections
            minSize=(20, 20),  # Smaller min face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Comment out face detection count for cleaner output
        # print(f"Haar detector found {len(faces)} faces")
        return faces

def preprocess_image_for_facenet(image_array):
    """
    Preprocesses an image (NumPy array) for the FaceNet model.
    Handles input from either OpenCV or PIL.
    Resizes to 160x160 and standardizes the image.
    """
    # If image is from PIL, it's already in RGB format
    # If input is a PIL Image object, convert to numpy array
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    # Check if array is valid
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty or invalid image array")
    
    # Resize to the expected input shape (160x160)
    try:
        resized_image = Image.fromarray(image_array).resize((160, 160))
    except Exception as e:
        print(f"Error during image resize: {e}")
        print(f"Image array shape: {image_array.shape}, dtype: {image_array.dtype}")
        raise
    
    # Convert back to numpy array
    img = np.array(resized_image)
    
    # Convert to float32
    img = img.astype('float32')
    
    # Standardize to [-1,1]
    img /= 255.0
    img = (img - 0.5) * 2.0
    
    # Add batch dimension (expected by model.predict)
    img = np.expand_dims(img, axis=0)
    
    return img

def extract_face(image_array, face_box, margin=20):
    """
    Extract a face from an image with a margin around the bounding box.
    
    Args:
        image_array: The image as a numpy array
        face_box: The face bounding box (x, y, w, h)
        margin: Additional margin around the face (pixels)
        
    Returns:
        The extracted face as a numpy array or None if extraction fails
    """
    x, y, w, h = face_box
    
    # Validate face box dimensions
    if w <= 0 or h <= 0:
        print(f"Invalid face box dimensions: w={w}, h={h}")
        return None
    
    # Calculate coordinates with margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image_array.shape[1], x + w + margin)
    y2 = min(image_array.shape[0], y + h + margin)
    
    # Validate coordinates
    if x1 >= x2 or y1 >= y2:
        # Comment out debug message for cleaner output
        # print(f"Invalid face coordinates after margin adjustment: ({x1},{y1},{x2},{y2})")
        return None
    
    # Extract face region
    face = image_array[y1:y2, x1:x2]
    
    # Validate extracted face
    if face is None or face.size == 0:
        # Comment out debug message for cleaner output
        # print(f"Extracted face region is empty")
        return None
        
    # Make sure we have a 3-channel RGB image
    if len(face.shape) < 3 or face.shape[2] < 3:
        # Comment out debug message for cleaner output
        # print(f"Converting grayscale face to RGB")
        # Convert grayscale to RGB if needed
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        else:
            # If it's some other weird format, return None
            return None
    
    return face

def save_debug_image(image, faces, output_path):
    """
    Saves a debug image with face bounding boxes drawn on it.
    
    Args:
        image: The original image
        faces: List of face bounding boxes (x, y, w, h)
        output_path: Path to save the debug image
    """
    # Make a copy to avoid modifying the original
    debug_img = image.copy()
    
    # Draw each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save the image
    cv2.imwrite(output_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    # Comment out debug message for cleaner output
    # print(f"Debug image saved to {output_path}")