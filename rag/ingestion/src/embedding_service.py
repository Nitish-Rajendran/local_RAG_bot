import numpy as np
from typing import List, Dict, Any
import logging

class EmbeddingService:
    """Generates embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Convert to list of numpy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = [embeddings[i] for i in range(embeddings.shape[0])]
            
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode([text])
            return embedding[0]
        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}")
            return np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if self.model is None:
            return 0
        
        # Generate a test embedding to get the dimension
        test_embedding = self.generate_single_embedding("test")
        return len(test_embedding)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, document_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate similarities between query and multiple document embeddings"""
        similarities = []
        
        for doc_embedding in document_embeddings:
            similarity = self.similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'model_loaded': self.model is not None
        } 