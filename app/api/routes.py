from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from app.models.schemas import PostResponse, SearchResponse, PostMetadata
from app.services.clip_service import CLIPService
from app.services.whisper_service import WhisperService
from app.services.video_service import VideoService
from app.vectors.faiss_client import FAISSClient
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()
clip_service = CLIPService()
whisper_service = WhisperService()
video_service = VideoService()
faiss_client = FAISSClient()

@router.post("/classify", response_model=PostResponse)
async def classify_post(
    images: Optional[List[UploadFile]] = File(None),
    videos: Optional[List[UploadFile]] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Classify a social media post using image, video, and/or text.
    Returns the post's embedding and metadata.
    """
    logger.info(f"Classify request received - images: {len(images) if images else 0}, videos: {len(videos) if videos else 0}, text: {bool(text)}")
    if not images and not videos and not text:
        raise HTTPException(status_code=400, detail="At least one of image, video, or text is required.")
    try:
        # Process videos if provided
        video_frames = []
        audio_text = None
        if videos:
            for video in videos:
                video_frames.extend(await video_service.extract_frames(video))
                if not audio_text:
                    audio_text = await whisper_service.transcribe(video)
        
        # Generate embeddings
        embeddings = await clip_service.generate_embeddings(
            images=images,
            video_frames=video_frames,
            text=text or audio_text
        )
        
        # Store in FAISS
        post_id = await faiss_client.add_post(embeddings, {
            "text": text,
            "audio_text": audio_text,
            "has_image": bool(images),
            "has_video": bool(videos)
        })
        
        return PostResponse(
            post_id=post_id,
            embedding=embeddings.tolist(),
            metadata=PostMetadata(
                text=text,
                audio_text=audio_text,
                has_image=bool(images),
                has_video=bool(videos)
            )
        )
    except Exception as e:
        logger.error("Error during classification", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to classify post")

@router.get("/search", response_model=List[SearchResponse])
async def search_posts(query: str, limit: int = 10):
    """
    Search for posts using a text query.
    Returns the most similar posts with their metadata.
    """
    logger.info(f"Search request received - query: '{query}', limit: {limit}")
    try:
        # Generate query embedding
        query_embedding = await clip_service.generate_text_embedding(query)
        
        # Search in FAISS
        results = await faiss_client.search(query_embedding, limit)
        
        return [
            SearchResponse(
                post_id=result.id,
                score=result.score,
                metadata=result.metadata
            )
            for result in results
        ]
    except Exception as e:
        logger.error("Error during search", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform search")