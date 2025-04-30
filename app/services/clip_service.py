import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Optional
from fastapi import UploadFile
from app.core.config import settings
from app.utils.helpers import load_image_from_upload

class CLIPService:
    def __init__(self):
        self.device = settings.DEVICE
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.CLIP_MODEL_NAME,
            pretrained=settings.CLIP_MODEL_PRETRAINED,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)

    async def generate_embeddings(
        self,
        images: Optional[List[UploadFile]] = None,
        video_frames: Optional[List[Image.Image]] = None,
        text: Optional[str] = None
    ) -> np.ndarray:
        """Generate a unified embedding from image, video frames, and/or text."""
        embeddings = []
        
        # Process images if provided
        if images:
            image_embeddings = []
            for image in images:
                img = await load_image_from_upload(image)
                img = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_embedding = self.model.encode_image(img)
                image_embeddings.append(img_embedding.cpu().numpy())
            if image_embeddings:
                embeddings.append(np.mean(image_embeddings, axis=0))

        # Process video frames
        if video_frames:
            frame_embeddings = []
            for frame in video_frames:
                frame = self.preprocess(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    frame_embedding = self.model.encode_image(frame)
                frame_embeddings.append(frame_embedding.cpu().numpy())
            # Average frame embeddings
            if frame_embeddings:
                embeddings.append(np.mean(frame_embeddings, axis=0))

        # Process text
        if text:
            text_embedding = await self.generate_text_embedding(text)
            embeddings.append(text_embedding)

        # Combine all embeddings
        if not embeddings:
            raise ValueError("No valid input provided for embedding generation")
        
        def normalize(vec: np.ndarray) -> np.ndarray:
            return vec / np.linalg.norm(vec)

        return normalize(np.mean(embeddings, axis=0))

    async def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text input."""
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(tokens)
        return text_embedding.cpu().numpy()