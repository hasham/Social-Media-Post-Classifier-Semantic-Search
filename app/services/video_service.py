import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from typing import List
from fastapi import UploadFile
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VideoService:
    def __init__(self):
        self.frame_rate = settings.FRAME_SAMPLE_RATE
        self.max_duration = settings.MAX_VIDEO_DURATION

    async def extract_frames(self, video_file: UploadFile) -> List[Image.Image]:
        """
        Extract frames from video at specified intervals.
        Returns a list of PIL Images.
        """
        logger.info(f"Starting frame extraction from video: {video_file.filename}")
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            content = await video_file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # Open video file
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                raise ValueError("Invalid video: FPS is zero")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Limit duration
            if duration > self.max_duration:
                total_frames = int(self.max_duration * fps)

            # Calculate frame interval
            frame_interval = int(fps / self.frame_rate)
            frames = []
            skipped = 0

            # Extract frames
            for frame_idx in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if np.std(frame_rgb) < 5:
                    skipped += 1
                    continue
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

            cap.release()
            logger.info(f"Extracted {len(frames)} frames, skipped {skipped} low-variation frames.")
            return frames

        finally:
            # Clean up temporary file
            Path(temp_video_path).unlink(missing_ok=True)