import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import uuid

class VectorStore:
    """Manages vector database operations using ChromaDB"""
     # CHANGE THE PATH OF YOUR local folder BELOW
    def __init__(self, db_path: str = 'C:/local_RAG_bot/local/vector_db', collection_name: str = 'rag_documents'):
        self.db_path = db_path
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create database directory if it doesn't exist
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Connected to existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
                self.logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def store_documents(self, chunks: List[Dict[str, Any]], embeddings: List, batch_size: int = 5000) -> int:
        if not chunks or not embeddings:
            self.logger.warning("No chunks or embeddings to store")
            return 0

        try:
            total_stored = 0
            for start in range(0, len(chunks), batch_size):
                end = start + batch_size
                batch_chunks = chunks[start:end]
                batch_embeddings = embeddings[start:end]

                documents = []
                metadatas = []
                ids = []

                for i, chunk in enumerate(batch_chunks):
                    if i < len(batch_embeddings):
                        chunk_id = str(uuid.uuid4())
                        documents.append(chunk['content'])
                        metadatas.append({
                            'source': chunk.get('file_name', 'unknown'),
                            'chunk_id': chunk.get('chunk_index', i),
                            'file_path': chunk.get('file_path', ''),
                            'file_type': chunk.get('metadata', {}).get('file_type', 'unknown'),
                            'chunk_size': chunk.get('chunk_size', 0)
                        })
                        ids.append(chunk_id)

                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                total_stored += len(documents)

            self.logger.info(f"Stored {total_stored} document chunks in vector database")
            return total_stored

        except Exception as e:
            self.logger.error(f"Error storing documents: {e}")
            return 0
    
    def search(self, query_embedding, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding"""
        try:
            # Convert embedding to list if it's a numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            self.logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching vector database: {e}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using text query (will generate embedding internally)"""
        try:
            # Search using text query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching by text: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'database_path': self.db_path
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {'name': self.collection_name, 'count': 0, 'database_path': self.db_path}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
    
    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents from a specific source"""
        try:
            # Get all documents from the source
            results = self.collection.get(
                where={'source': source},
                include=['metadatas', 'ids']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                self.logger.info(f"Deleted {len(results['ids'])} documents from source: {source}")
                return len(results['ids'])
            else:
                self.logger.info(f"No documents found for source: {source}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents']:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting document by ID: {e}")
            return None 