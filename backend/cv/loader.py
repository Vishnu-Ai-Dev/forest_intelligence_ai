"""
Image loading and validation utilities for Forest Intelligence CV module.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def validate_and_load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Validates the input image path and loads the image as an RGB NumPy array.

    Args:
        image_path: Path to the image file (relative or absolute).

    Returns:
        np.ndarray: Image data of shape (H, W, 3) and dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path is invalid, extension unsupported, file empty, or corrupt.
    """
    if not image_path:
        raise ValueError("Image path cannot be empty")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    if not path.is_file():
        raise ValueError(f"Path points to a directory, not a file: {image_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    if path.stat().st_size == 0:
        raise ValueError("Image file is empty (0 bytes)")

    try:
        with Image.open(path) as img:
            # Force loading the image data to detect truncation or corruption
            img.load()
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)

            if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
                raise ValueError(f"Invalid image array shape: {arr.shape}")

            return arr
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError(f"Corrupt or unreadable image file '{image_path}': {str(err)}") from err
