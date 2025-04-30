import faiss
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path
from app.core.config import settings

class FAISSClient:
    def __init__(self):
        self.dimension = settings.VECTOR_DIMENSION
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = self.index_path.with_suffix('.json')
        
        # Initialize or load index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}

    async def add_post(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Add a post embedding and its metadata to the index."""
        # Generate unique ID
        post_id = str(len(self.metadata))
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Store metadata
        self.metadata[post_id] = metadata
        
        # Save to disk
        self._save()
        
        return post_id

    async def search(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        """Search for similar posts using a query embedding."""
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(limit, self.index.ntotal)
        )
        
        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                post_id = str(idx)
                results.append({
                    'id': post_id,
                    'score': float(1 / (1 + distance)),  # Convert distance to similarity score
                    'metadata': self.metadata[post_id]
                })
        
        return results

    def _save(self):
        """Save the index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f) 