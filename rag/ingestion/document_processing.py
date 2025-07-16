import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from document_loader import DocumentLoader
from text_chunker import TextChunker
from embedding_service import EmbeddingService
from vector_store import VectorStore
from metadata_manager import MetadataManager

class DocumentProcessingPipeline:
    """Main pipeline for processing documents into searchable knowledge base"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the document processing pipeline"""
        self.config = config or self._get_default_config()
        self.setup_logging()
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        self.embedding_service = EmbeddingService(
            model_name=self.config['embedding_model']
        )
        self.vector_store = VectorStore(
            db_path=self.config['vector_db_path'],
            collection_name=self.config['collection_name']
        )
        self.metadata_manager = MetadataManager()
        
        self.logger.info("Document Processing Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_dir': 'D:/rag_system/data',
            'vector_db_path': 'D:/rag_system/vector_db',
            'collection_name': 'rag_documents',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'supported_formats': ['.pdf', '.txt', '.docx', '.pptx', '.md']
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_documents(self, documents_dir: str = None) -> Dict[str, Any]:
        """Process all documents in the specified directory"""
        documents_dir = documents_dir or self.config['data_dir']
        
        self.logger.info(f"Starting document processing from: {documents_dir}")
        
        # Step 1: Load documents
        self.logger.info("Step 1: Loading documents...")
        documents = self.document_loader.load_documents(documents_dir)
        
        if not documents:
            self.logger.warning("No documents found to process")
            return {'status': 'no_documents', 'processed': 0}
        
        # Step 2: Chunk documents
        self.logger.info("Step 2: Chunking documents...")
        chunks = self.text_chunker.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        self.logger.info("Step 3: Generating embeddings...")
        embeddings = self.embedding_service.generate_embeddings(chunks)
        
        # Step 4: Store in vector database
        self.logger.info("Step 4: Storing in vector database...")
        stored_count = self.vector_store.store_documents(chunks, embeddings)
        
        # Step 5: Update metadata
        self.logger.info("Step 5: Updating metadata...")
        self.metadata_manager.update_metadata(documents, stored_count)
        
        self.logger.info(f"Document processing complete. Processed {stored_count} chunks")
        
        return {
            'status': 'success',
            'documents_loaded': len(documents),
            'chunks_created': len(chunks),
            'embeddings_generated': len(embeddings),
            'stored_in_db': stored_count
        }
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document"""
        self.logger.info(f"Processing single document: {file_path}")
        
        # Load single document
        document = self.document_loader.load_single_document(file_path)
        if not document:
            return {'status': 'error', 'message': 'Could not load document'}
        
        # Process the document
        chunks = self.text_chunker.chunk_documents([document])
        embeddings = self.embedding_service.generate_embeddings(chunks)
        stored_count = self.vector_store.store_documents(chunks, embeddings)
        
        return {
            'status': 'success',
            'document': file_path,
            'chunks_created': len(chunks),
            'stored_in_db': stored_count
        }
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about processed documents"""
        return self.vector_store.get_collection_info()
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using the vector database"""
        self.logger.info(f"Searching for: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embeddings([query])
        
        # Search vector database
        results = self.vector_store.search(query_embedding[0], top_k)
        
        return results
    
    def clear_database(self):
        """Clear all documents from the vector database"""
        self.logger.info("Clearing vector database...")
        self.vector_store.clear_collection()
        self.metadata_manager.clear_metadata()
        self.logger.info("Vector database cleared")

def main():
    """Main function to run the document processing pipeline"""
    print("🚀 Module 2: Document Processing Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline()
    
    # Check if documents exist
    data_dir = pipeline.config['data_dir']
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Please create the directory and add some documents")
        return
    
    # Process documents
    result = pipeline.process_documents()
    
    if result['status'] == 'success':
        print(f"Successfully processed {result['stored_in_db']} document chunks")
        print(f"Documents loaded: {result['documents_loaded']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Embeddings generated: {result['embeddings_generated']}")
    else:
        print(f"Processing failed: {result}")
    
    # Show collection info
    info = pipeline.get_document_info()
    print(f"\n  Vector Database Info:")
    print(f"   Collection: {info.get('name', 'N/A')}")
    print(f"   Document count: {info.get('count', 0)}")

if __name__ == "__main__":
    main() 