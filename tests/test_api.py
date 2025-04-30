from fastapi.testclient import TestClient
from main import app
import pytest
from pathlib import Path
import os

client = TestClient(app)

def test_search_endpoint():
    """Test the search endpoint with a simple query."""
    response = client.get("/api/search?query=test")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_classify_endpoint_no_input():
    """Test the classify endpoint with no input."""
    response = client.post("/api/classify")
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_classify_endpoint_with_image():
    """Test the classify endpoint with an image."""
    # Create a test image
    test_image = Path("tests/data/test_image.jpg")
    test_image.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test image
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(test_image)
    
    try:
        with open(test_image, "rb") as f:
            response = client.post(
                "/api/classify",
                files={"image": ("test_image.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        assert "post_id" in response.json()
        assert "embedding" in response.json()
    finally:
        # Clean up test image
        if test_image.exists():
            test_image.unlink() 