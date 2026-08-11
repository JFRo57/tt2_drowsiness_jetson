import cv2
import numpy as np

def preprocess_eye(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image crop of single eye for Model A (eye_state.onnx).
    Input: BGR/RGB or Grayscale numpy image array of single eye.
    Output: Normalized float32 numpy array with shape (1, 1, 64, 64) in range [0.0, 1.0].
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0) # (1, 1, 64, 64)
    return tensor


def preprocess_yawn(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image crop of mouth for Model B (yawn_state.onnx).
    Input: BGR/RGB or Grayscale numpy image array of mouth.
    Output: Normalized float32 numpy array with shape (1, 1, 64, 96) in range [0.0, 1.0].
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    resized = cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0) # (1, 1, 64, 96)
    return tensor


def preprocess_head_pose(image: np.ndarray) -> np.ndarray:
    """
    Preprocess aligned face image for Model C (head_pose.onnx).
    Input: BGR numpy image array of face.
    Output: Normalized RGB float32 numpy array with shape (1, 3, 128, 128) in range [0.0, 1.0].
    """
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image.copy()

    resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1)) # (3, 128, 128)
    tensor = np.expand_dims(chw, axis=0)      # (1, 3, 128, 128)
    return tensor
