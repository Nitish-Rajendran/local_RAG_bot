import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ingestion', 'src'))

from embedding_service import EmbeddingService
from vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    content: str
    file_name: str
    chunk_index: int
    similarity_score: float
    metadata: Dict
    source_path: str


class RetrievalEngine:
    """
    Engine for retrieving relevant documents using vector similarity search.
    
    Features:
    - Multi-query retrieval
    - Relevance scoring
    - Result ranking and filtering
    - Context-aware retrieval
    """
     # CHANGE THE PATH OF YOUR local folder BELOW
    def __init__(self, vector_store_path: str = "C:/local_RAG_bot/local/vector_db", 
                 collection_name: str = "documents"):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_store_path: Path to the vector database
            collection_name: Name of the collection to search
        """
        self.vector_store = VectorStore(db_path=vector_store_path, 
                                      collection_name=collection_name)
        self.embedding_service = EmbeddingService()
        self.logger = logging.getLogger(__name__)
        
        # Retrieval parameters
        self.default_top_k = 5
        self.min_similarity_threshold = 0.3
        self.max_context_length = 2000  # characters
        
    def search_single_query(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Search for relevant documents using a single query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of retrieval results
        """
        if top_k is None:
            top_k = self.default_top_k
            
        try:
            # Search in vector store
            search_results = self.vector_store.search_by_text(query, top_k=top_k)
            
            if not search_results:
                self.logger.warning(f"No results found for query: {query}")
                return []
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.get('content', ''),
                    file_name=result.get('file_name', ''),
                    chunk_index=result.get('chunk_index', 0),
                    similarity_score=result.get('similarity_score', 0.0),
                    metadata=result.get('metadata', {}),
                    source_path=result.get('file_path', '')
                )
                results.append(retrieval_result)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= self.min_similarity_threshold
            ]
            
            self.logger.info(f"Retrieved {len(filtered_results)} relevant results for query: {query}")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in search_single_query: {e}")
            return []
    
    def search_multiple_queries(self, queries: List[str], top_k: int = None) -> List[RetrievalResult]:
        """
        Search using multiple query variations and combine results.
        
        Args:
            queries: List of query variations
            top_k: Number of top results per query
            
        Returns:
            Combined and deduplicated retrieval results
        """
        all_results = []
        
        for query in queries:
            results = self.search_single_query(query, top_k)
            all_results.extend(results)
        
        # Deduplicate results based on content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_results
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Remove duplicate or very similar results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Create a simplified content hash for comparison
            content_hash = self._create_content_hash(result.content)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _create_content_hash(self, content: str) -> str:
        """
        Create a simplified hash of content for deduplication.
        
        Args:
            content: Document content
            
        Returns:
            Content hash string
        """
        # Take first 100 characters and last 100 characters
        if len(content) <= 200:
            return content.lower().strip()
        else:
            return (content[:100] + content[-100:]).lower().strip()
    
    def get_context_window(self, results: List[RetrievalResult], 
                          max_length: int = None) -> str:
        """
        Build a context window from retrieval results.
        
        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
            
        Returns:
            Combined context string
        """
        if max_length is None:
            max_length = self.max_context_length
        
        context_parts = []
        current_length = 0
        
        for result in results:
            # Add file name as header
            header = f"[From: {result.file_name}]"
            content = f"{header}\n{result.content}\n"
            
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Truncate if needed
                remaining_length = max_length - current_length
                if remaining_length > 50:  # Only add if we have meaningful space
                    truncated_content = content[:remaining_length] + "..."
                    context_parts.append(truncated_content)
                break
        
        return "\n".join(context_parts)
    
    def calculate_relevance_score(self, query: str, result: RetrievalResult) -> float:
        """
        Calculate a custom relevance score for a result.
        
        Args:
            query: Original query
            result: Retrieval result
            
        Returns:
            Relevance score between 0 and 1
        """
        # Base score from vector similarity
        base_score = result.similarity_score
        
        # Keyword matching bonus
        query_words = set(query.lower().split())
        content_words = set(result.content.lower().split())
        keyword_overlap = len(query_words.intersection(content_words))
        keyword_bonus = min(keyword_overlap * 0.1, 0.3)
        
        # Content length bonus (prefer longer, more detailed content)
        length_bonus = min(len(result.content) / 1000, 0.2)
        
        # Combine scores
        final_score = base_score + keyword_bonus + length_bonus
        return min(final_score, 1.0)
    
    def get_retrieval_statistics(self, results: List[RetrievalResult]) -> Dict:
        """
        Get statistics about the retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                'total_results': 0,
                'average_similarity': 0.0,
                'unique_sources': 0,
                'total_content_length': 0
            }
        
        similarities = [result.similarity_score for result in results]
        sources = set(result.file_name for result in results)
        content_length = sum(len(result.content) for result in results)
        
        return {
            'total_results': len(results),
            'average_similarity': np.mean(similarities),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'unique_sources': len(sources),
            'total_content_length': content_length,
            'average_content_length': content_length / len(results)
        }
    
    def search_with_filters(self, query: str, filters: Dict = None, 
                           top_k: int = None) -> List[RetrievalResult]:
        """
        Search with additional filters (file type, date range, etc.).
        
        Args:
            query: Search query
            filters: Dictionary of filters to apply
            top_k: Number of top results
            
        Returns:
            Filtered retrieval results
        """
        results = self.search_single_query(query, top_k)
        
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            # Apply filters
            include_result = True
            
            # File type filter
            if 'file_type' in filters:
                file_extension = result.file_name.split('.')[-1].lower()
                if file_extension not in filters['file_type']:
                    include_result = False
            
            # Source filter
            if 'sources' in filters:
                if result.file_name not in filters['sources']:
                    include_result = False
            
            # Similarity threshold filter
            if 'min_similarity' in filters:
                if result.similarity_score < filters['min_similarity']:
                    include_result = False
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retrieval engine
    engine = RetrievalEngine()
    
    # Test queries TEST WITH YOUR CUSTOM QUERIES BASED ON YOUR DOCUMENTS IN data FOLDER
    test_queries = [
        "What is python classes?",
        "How to import a python package?",
        "Explain inheritance in python?"
    ]
    
    print(" Testing Retrieval Engine")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n Query: {query}")
        
        # Single query search
        results = engine.search_single_query(query, top_k=3)
        
        if results:
            print(f"    Found {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show top 2
                print(f"    Result {i+1}: {result.file_name} (score: {result.similarity_score:.3f})")
                print(f"      Content preview: {result.content[:100]}...")
        else:
            print("    No results found")
        
        # Get statistics
        stats = engine.get_retrieval_statistics(results)
        print(f"    Average similarity: {stats['average_similarity']:.3f}")
        print(f"    Unique sources: {stats['unique_sources']}") 