from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Social Media Post Classifier"
    
    # Model Settings
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_MODEL_PRETRAINED: str = "openai"
    WHISPER_MODEL_SIZE: str = "base"  # or "tiny" for faster processing
    
    # Vector DB Settings
    VECTOR_DIMENSION: int = 512  # CLIP embedding dimension
    FAISS_INDEX_PATH: Path = Path("data/faiss_index")
    
    # Media Processing
    MAX_VIDEO_DURATION: int = 300  # 5 minutes
    FRAME_SAMPLE_RATE: int = 1  # Extract 1 frame per second
    TEMP_DIR: Path = Path("data/temp")
    
    # Hardware Settings
    DEVICE: str = "mps" if os.getenv("USE_MPS", "true").lower() == "true" else "cpu"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Create necessary directories
settings.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True) 