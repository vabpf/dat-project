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

def represent_with_custom_model(
    img_path: Union[str, np.ndarray, IO[bytes]],
    custom_model: Model,
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    normalization: str = "base",
    enforce_face_detection = False
) -> List[Dict[str, Any]]:
    resp_objs = []
    model = custom_model
    target_size = model.input_shape[1:3]

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_face_detection,
        align=align,
    )

    for img_obj in img_objs:
        img = img_obj["face"]

        # Preprocessing steps
        img = preprocessing.resize_image(img, target_size)
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        # --------------------------- THE FIX ---------------------------
        # The input `img` might be 3D or 4D. We must guarantee it becomes
        # a 4D tensor of shape (1, 112, 112, 3) before prediction.
        # `reshape` is the most robust way to do this.

        # Reshape the image to a 4D batch with a single item
        img_batch = np.reshape(img, (1, *target_size, 3))

        # Perform prediction on the correctly shaped batch
        embedding = model.predict(img_batch, verbose=0)[0].tolist()
        # ---------------------------------------------------------------

        resp_objs.append({
            "embedding": embedding,
            "facial_area": img_obj["facial_area"],
            "face_confidence": img_obj["confidence"],
        })
    return resp_objs