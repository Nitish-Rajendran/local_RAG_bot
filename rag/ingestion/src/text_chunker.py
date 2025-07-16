import re
from typing import List, Dict, Any
import logging

class TextChunker:
    """Splits documents into chunks for better processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for document in documents:
            chunks = self.chunk_single_document(document)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def chunk_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document"""
        content = document.get('content', '')
        if not content:
            return []
        
        # Clean the text
        cleaned_text = self._clean_text(content)
        
        # Split into chunks
        text_chunks = self._split_text(cleaned_text)
        
        # Create chunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'id': f"{document['file_name']}_chunk_{i}",
                'content': chunk_text,
                'file_path': document['file_path'],
                'file_name': document['file_name'],
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text),
                'metadata': {
                    'source': document['file_name'],
                    'chunk_id': i,
                    'file_type': document.get('file_extension', 'unknown')
                }
            }
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks from {document['file_name']}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                
                # Find the last sentence ending in this range
                sentence_end = self._find_sentence_boundary(
                    text[search_start:search_end], 
                    search_start
                )
                
                if sentence_end > start + self.chunk_size * 0.8:  # Only use if it's not too early
                    end = sentence_end
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, offset: int) -> int:
        """Find the last sentence boundary in the given text"""
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        last_boundary = offset
        
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos != -1:
                last_boundary = max(last_boundary, offset + pos + len(ending))
        
        return last_boundary
    
    def chunk_by_sentences(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Alternative chunking method that respects sentence boundaries"""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {'total_chunks': 0, 'avg_chunk_size': 0, 'min_chunk_size': 0, 'max_chunk_size': 0}
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        } 