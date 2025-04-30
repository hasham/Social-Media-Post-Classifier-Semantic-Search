import asyncio
import argparse
from pathlib import Path
from app.services.clip_service import CLIPService
from app.services.whisper_service import WhisperService
from app.services.video_service import VideoService
from app.vectors.faiss_client import FAISSClient
from app.utils.helpers import load_image_from_upload
from fastapi import UploadFile
import json

async def process_post(
    image_path: Path = None,
    video_path: Path = None,
    text: str = None,
    clip_service: CLIPService = None,
    whisper_service: WhisperService = None,
    video_service: VideoService = None,
    faiss_client: FAISSClient = None
):
    """Process a single post and add it to the index."""
    # Process video if provided
    video_frames = []
    audio_text = None
    if video_path:
        with open(video_path, "rb") as f:
            video_file = UploadFile(filename=video_path.name, file=f)
            video_frames = await video_service.extract_frames(video_file)
            audio_text = await whisper_service.transcribe(video_file)

    # Process image if provided
    image = None
    if image_path:
        with open(image_path, "rb") as f:
            image = UploadFile(filename=image_path.name, file=f)

    # Generate embeddings
    embeddings = await clip_service.generate_embeddings(
        image=image,
        video_frames=video_frames,
        text=text or audio_text
    )

    # Store in FAISS
    post_id = await faiss_client.add_post(embeddings, {
        "text": text,
        "audio_text": audio_text,
        "has_image": bool(image),
        "has_video": bool(video_path)
    })

    return post_id

async def main():
    parser = argparse.ArgumentParser(description="Process demo data for social media post classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing demo data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Initialize services
    clip_service = CLIPService()
    whisper_service = WhisperService()
    video_service = VideoService()
    faiss_client = FAISSClient()

    # Process all posts in the data directory
    for post_dir in data_dir.iterdir():
        if not post_dir.is_dir():
            continue

        print(f"Processing post in {post_dir}")
        
        # Load post metadata
        metadata_path = post_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Process post
        image_path = next(post_dir.glob("*.jpg"), None)
        video_path = next(post_dir.glob("*.mp4"), None)
        text = metadata.get("text")

        try:
            post_id = await process_post(
                image_path=image_path,
                video_path=video_path,
                text=text,
                clip_service=clip_service,
                whisper_service=whisper_service,
                video_service=video_service,
                faiss_client=faiss_client
            )
            print(f"Successfully processed post {post_id}")
        except Exception as e:
            print(f"Error processing post in {post_dir}: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 