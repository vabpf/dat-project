import os
import json
import numpy as np
from tqdm import tqdm
import logging
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from deepface import DeepFace
from deepface.commons import image_utils
from deepface.modules import detection, preprocessing
from typing import Any, Dict, List, Union, IO

MODEL_NAME= "ArcFace"

def load_custom_arcface_model(weights_path: str) -> Model:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    print(f"Loading base '{MODEL_NAME}' model structure with DeepFace...")
    try:
        # Attempt to build the model using DeepFace's internal logic for ArcFace
        # This might handle potential version or structure differences
        base_model_wrapper = DeepFace.build_model(MODEL_NAME)
        base_model = base_model_wrapper.model # Get the Keras Model object

        print(f"Base '{MODEL_NAME}' model structure loaded successfully.")
        print("Base model summary:")

        print(f"Loading custom weights from {weights_path}...")
        try:
            base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("Custom weights loaded successfully using by_name=True, skip_mismatch=True.")
        except Exception as e:
            print(f"Error loading weights with by_name=True: {e}. Attempting strict loading...")
            try:
                base_model.load_weights(weights_path)
                print("Custom weights loaded successfully using strict loading.")
            except Exception as e_strict:
                raise RuntimeError(f"Failed to load custom weights from {weights_path}: {e_strict}")

    except Exception as e_build:
        raise RuntimeError(f"Failed to build base '{MODEL_NAME}' model structure with DeepFace: {e_build}")

    return base_model