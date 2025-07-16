import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

class MetadataManager:
    """Manages metadata about processed documents"""
    
    # CHANGE THE PATH OF YOUR local folder  BELOW
    def __init__(self, metadata_file: str = 'C:/local_RAG_bot/local/data/metadata.json'):
        self.metadata_file = metadata_file
        self.logger = logging.getLogger(__name__)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        try:
            if Path(self.metadata_file).exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    'documents': {},
                    'processing_history': [],
                    'statistics': {
                        'total_documents': 0,
                        'total_chunks': 0,
                        'total_size': 0,
                        'last_processed': None
                    }
                }
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return {
                'documents': {},
                'processing_history': [],
                'statistics': {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'total_size': 0,
                    'last_processed': None
                }
            }
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            # Create directory if it doesn't exist
            Path(self.metadata_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def update_metadata(self, documents: List[Dict[str, Any]], chunks_processed: int):
        """Update metadata with new document processing information"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Update document information
            for doc in documents:
                file_path = doc['file_path']
                file_name = doc['file_name']
                
                self.metadata['documents'][file_path] = {
                    'file_name': file_name,
                    'file_size': doc.get('size', 0),
                    'file_type': doc.get('file_extension', 'unknown'),
                    'content_length': len(doc.get('content', '')),
                    'processed_at': timestamp,
                    'chunks_created': chunks_processed // len(documents) if documents else 0
                }
            
            # Update processing history
            processing_record = {
                'timestamp': timestamp,
                'documents_processed': len(documents),
                'chunks_created': chunks_processed,
                'files': [doc['file_name'] for doc in documents]
            }
            self.metadata['processing_history'].append(processing_record)
            
            # Update statistics
            self.metadata['statistics']['total_documents'] = len(self.metadata['documents'])
            self.metadata['statistics']['total_chunks'] += chunks_processed
            self.metadata['statistics']['total_size'] += sum(doc.get('size', 0) for doc in documents)
            self.metadata['statistics']['last_processed'] = timestamp
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Updated metadata for {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a specific document"""
        return self.metadata['documents'].get(file_path, {})
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get information about all processed documents"""
        return self.metadata['documents']
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing history"""
        return self.metadata['processing_history'][-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.metadata['statistics']
    
    def remove_document(self, file_path: str) -> bool:
        """Remove a document from metadata"""
        try:
            if file_path in self.metadata['documents']:
                # Update statistics
                doc_info = self.metadata['documents'][file_path]
                self.metadata['statistics']['total_documents'] -= 1
                self.metadata['statistics']['total_size'] -= doc_info.get('file_size', 0)
                
                # Remove document
                del self.metadata['documents'][file_path]
                
                # Save metadata
                self._save_metadata()
                
                self.logger.info(f"Removed document from metadata: {file_path}")
                return True
            else:
                self.logger.warning(f"Document not found in metadata: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing document from metadata: {e}")
            return False
    
    def clear_metadata(self):
        """Clear all metadata"""
        try:
            self.metadata = {
                'documents': {},
                'processing_history': [],
                'statistics': {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'total_size': 0,
                    'last_processed': None
                }
            }
            self._save_metadata()
            self.logger.info("Cleared all metadata")
            
        except Exception as e:
            self.logger.error(f"Error clearing metadata: {e}")
    
    def export_metadata(self, export_file: str) -> bool:
        """Export metadata to a file"""
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported metadata to: {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {e}")
            return False
    
    def get_documents_by_type(self, file_type: str) -> List[Dict[str, Any]]:
        """Get all documents of a specific type"""
        documents = []
        for file_path, doc_info in self.metadata['documents'].items():
            if doc_info.get('file_type', '').lower() == file_type.lower():
                documents.append({
                    'file_path': file_path,
                    **doc_info
                })
        return documents
    
    def get_documents_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get documents processed within a date range"""
        documents = []
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        for file_path, doc_info in self.metadata['documents'].items():
            processed_at = datetime.fromisoformat(doc_info.get('processed_at', ''))
            if start_dt <= processed_at <= end_dt:
                documents.append({
                    'file_path': file_path,
                    **doc_info
                })
        
        return documents 