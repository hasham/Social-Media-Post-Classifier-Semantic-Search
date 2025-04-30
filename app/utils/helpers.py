from PIL import Image
import io
from fastapi import UploadFile
import numpy as np
from typing import List

async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load an image from an uploaded file."""
    content = await file.read()
    return Image.open(io.BytesIO(content))

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def merge_embeddings(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """Merge multiple embeddings by averaging."""
    if not embeddings_list:
        raise ValueError("No embeddings provided")
    return np.mean(embeddings_list, axis=0)

def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    return text.lower().strip() 