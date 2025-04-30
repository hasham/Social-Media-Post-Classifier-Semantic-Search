from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PostMetadata(BaseModel):
    text: Optional[str] = None
    audio_text: Optional[str] = None
    has_image: bool = False
    has_video: bool = False
    created_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class PostResponse(BaseModel):
    post_id: str
    embedding: List[float]
    metadata: PostMetadata  # Use structured metadata

class SearchResponse(BaseModel):
    post_id: str
    score: float = Field(..., description="Similarity score (higher is better)")
    metadata: PostMetadata  # Also structured for consistency