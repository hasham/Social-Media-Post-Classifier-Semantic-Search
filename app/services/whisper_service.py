import logging
import whisper
import tempfile
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class WhisperService:
    def __init__(self):
        self.model = whisper.load_model(settings.WHISPER_MODEL_SIZE)
        self.device = settings.DEVICE

    async def transcribe(self, video_file: UploadFile) -> str:
        """
        Extract audio from video and transcribe it using Whisper.
        Returns the transcribed text.
        """
        logger.info(f"Received video file for transcription: {video_file.filename}")
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            content = await video_file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # Transcribe audio
            result = self.model.transcribe(
                temp_video_path,
                language="en",  # Can be made configurable
                fp16=False if self.device == "cpu" else True
            )
            logger.info(f"Transcription result (preview): {result['text'][:100]}...")
            return result["text"].strip()
        finally:
            # Clean up temporary file
            Path(temp_video_path).unlink(missing_ok=True) 