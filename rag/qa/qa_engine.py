import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from rag.retrieval.query_processor import QueryProcessor
from rag.retrieval.retrieval_engine import RetrievalEngine
from rag.retrieval.context_builder import ContextBuilder
from rag.qa.answer_generator import AnswerGenerator, GeneratedAnswer
from response_formatter import ResponseFormatter, FormattedResponse


@dataclass
class QAEngineConfig:
    """Configuration for the QA Engine."""
     # CHANGE THE PATH OF YOUR "vector_db" folder INSIDE local folder
    vector_store_path: str = "C:/local_RAG_bot/local/vector_db"
    collection_name: str = "documents"
    ollama_url: str = "http://localhost:11434"
    model_name: str = "mistral:7b"
    max_context_length: int = 3000
    max_answer_length: int = 1000
    min_confidence_threshold: float = 0.3
    enable_logging: bool = True


@dataclass
class QAEngineResult:
    """Result from the QA Engine."""
    query: str
    answer: str
    confidence_score: float
    sources: List[str]
    processing_time: float
    context_used: str
    metadata: Dict
    formatted_response: FormattedResponse


class QAEngine:
    """
    Main Question-Answering Engine that orchestrates all components.
    
    Features:
    - Complete RAG pipeline
    - Query processing and understanding
    - Intelligent document retrieval
    - Context-aware answer generation
    - Response formatting and presentation
    """
    
    def __init__(self, config: QAEngineConfig = None):
        """
        Initialize the QA Engine.
        
        Args:
            config: Configuration for the QA Engine
        """
        # Always use mistral:7b and localhost endpoint
        self.config = config or QAEngineConfig()
        self.config.ollama_url = "http://localhost:11434"
        self.config.model_name = "mistral:7b"
        
        # Set up logging
        if self.config.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.retrieval_engine = RetrievalEngine(
            vector_store_path=self.config.vector_store_path,
            collection_name=self.config.collection_name
        )
        self.context_builder = ContextBuilder(
            max_context_length=self.config.max_context_length
        )
        self.answer_generator = AnswerGenerator(
            ollama_url=self.config.ollama_url,
            model_name=self.config.model_name
        )
        self.response_formatter = ResponseFormatter()
        
        self.logger.info("QA Engine initialized successfully")
    
    def ask_question(self, query: str) -> QAEngineResult:
        """
        Ask a question and get a comprehensive answer.
        
        Args:
            query: User's question
            
        Returns:
            Complete QA engine result
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Process the query
            processed_query = self.query_processor.process_query(query)
            self.logger.info(f"Query processed - Keywords: {processed_query['keywords']}")
            
            # Step 2: Retrieve relevant documents
            retrieval_results = self.retrieval_engine.search_multiple_queries(
                processed_query['query_variations']
            )
            self.logger.info(f"Retrieved {len(retrieval_results)} relevant documents")
            
            # Step 3: Build context
            context = self.context_builder.build_context(query, retrieval_results)
            self.logger.info(f"Built context with {len(context)} characters")
            
            # Step 4: Generate answer
            generated_answer = self.answer_generator.generate_answer(query, context)
            self.logger.info(f"Generated answer with confidence: {generated_answer.confidence_score:.3f}")
            
            # Step 5: Format response
            processing_time = time.time() - start_time
            formatted_response = self.response_formatter.format_response(
                generated_answer, query, processing_time
            )
            
            # Create result
            result = QAEngineResult(
                query=query,
                answer=generated_answer.answer,
                confidence_score=generated_answer.confidence_score,
                sources=generated_answer.sources_used,
                processing_time=processing_time,
                context_used=context,
                metadata=formatted_response.metadata,
                formatted_response=formatted_response
            )
            
            self.logger.info(f"Question answered successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return self._create_error_result(query, str(e), time.time() - start_time)
    
    def ask_question_simple(self, query: str) -> str:
        """
        Ask a question and get a simple text answer.
        
        Args:
            query: User's question
            
        Returns:
            Simple text answer
        """
        result = self.ask_question(query)
        return result.answer
    
    def ask_question_with_metadata(self, query: str) -> Dict:
        """
        Ask a question and get answer with full metadata.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        result = self.ask_question(query)
        return {
            'query': result.query,
            'answer': result.answer,
            'confidence': result.confidence_score,
            'sources': result.sources,
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }
    
    def batch_ask_questions(self, queries: List[str]) -> List[QAEngineResult]:
        """
        Process multiple questions in batch.
        
        Args:
            queries: List of questions
            
        Returns:
            List of QA engine results
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Processing question {i}/{len(queries)}: {query}")
            result = self.ask_question(query)
            results.append(result)
        
        return results
    
    def get_system_status(self) -> Dict:
        """
        Get the current status of the QA system.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'engine_initialized': True,
            'components': {
                'query_processor': True,
                'retrieval_engine': True,
                'context_builder': True,
                'answer_generator': True,
                'response_formatter': True
            },
            'configuration': {
                'vector_store_path': self.config.vector_store_path,
                'collection_name': self.config.collection_name,
                'ollama_url': self.config.ollama_url,
                'model_name': self.config.model_name
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Test component connectivity
        try:
            # Test retrieval engine
            test_results = self.retrieval_engine.search_single_query("test", top_k=1)
            status['components']['retrieval_engine'] = len(test_results) >= 0
        except Exception as e:
            status['components']['retrieval_engine'] = False
            status['retrieval_error'] = str(e)
        
        return status
    
    def validate_answer_quality(self, result: QAEngineResult) -> Tuple[bool, List[str]]:
        """
        Validate the quality of a generated answer.
        
        Args:
            result: QA engine result
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check confidence threshold
        if result.confidence_score < self.config.min_confidence_threshold:
            issues.append(f"Confidence too low: {result.confidence_score:.3f}")
        
        # Check answer length
        if len(result.answer) < 20:
            issues.append("Answer too short")
        elif len(result.answer) > self.config.max_answer_length:
            issues.append("Answer too long")
        
        # Check processing time
        if result.processing_time > 30:  # 30 seconds threshold
            issues.append("Processing time too high")
        
        # Check if sources were found
        if not result.sources:
            issues.append("No sources referenced")
        
        # Check context usage
        if len(result.context_used) < 100:
            issues.append("Limited context used")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_performance_metrics(self, results: List[QAEngineResult]) -> Dict:
        """
        Calculate performance metrics from multiple results.
        
        Args:
            results: List of QA engine results
            
        Returns:
            Dictionary with performance metrics
        """
        if not results:
            return {}
        
        processing_times = [r.processing_time for r in results]
        confidence_scores = [r.confidence_score for r in results]
        answer_lengths = [len(r.answer) for r in results]
        
        return {
            'total_questions': len(results),
            'average_processing_time': sum(processing_times) / len(processing_times),
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'average_answer_length': sum(answer_lengths) / len(answer_lengths),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'successful_answers': len([r for r in results if r.confidence_score > 0.5])
        }
    
    def _create_error_result(self, query: str, error_message: str, 
                           processing_time: float) -> QAEngineResult:
        """
        Create an error result when processing fails.
        
        Args:
            query: Original query
            error_message: Error message
            processing_time: Processing time
            
        Returns:
            Error result
        """
        error_answer = f"I apologize, but I encountered an error while processing your question: {error_message}"
        
        return QAEngineResult(
            query=query,
            answer=error_answer,
            confidence_score=0.0,
            sources=[],
            processing_time=processing_time,
            context_used="",
            metadata={
                'error': True,
                'error_message': error_message,
                'processing_time': processing_time
            },
            formatted_response=None
        )
    
    def export_results(self, results: List[QAEngineResult], 
                      format_type: str = "json") -> str:
        """
        Export results in different formats.
        
        Args:
            results: List of QA engine results
            format_type: Export format ('json', 'csv', 'text')
            
        Returns:
            Exported data as string
        """
        if format_type == "json":
            return self._export_json(results)
        elif format_type == "csv":
            return self._export_csv(results)
        elif format_type == "text":
            return self._export_text(results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_json(self, results: List[QAEngineResult]) -> str:
        """Export results as JSON."""
        import json
        
        export_data = []
        for result in results:
            export_data.append({
                'query': result.query,
                'answer': result.answer,
                'confidence': result.confidence_score,
                'sources': result.sources,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            })
        
        return json.dumps(export_data, indent=2)
    
    def _export_csv(self, results: List[QAEngineResult]) -> str:
        """Export results as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Query', 'Answer', 'Confidence', 'Sources', 'Processing Time'])
        
        # Write data
        for result in results:
            sources_str = '; '.join(result.sources) if result.sources else 'None'
            writer.writerow([
                result.query,
                result.answer,
                f"{result.confidence_score:.3f}",
                sources_str,
                f"{result.processing_time:.2f}"
            ])
        
        return output.getvalue()
    
    def _export_text(self, results: List[QAEngineResult]) -> str:
        """Export results as formatted text."""
        text_parts = []
        
        for i, result in enumerate(results, 1):
            text_parts.append(f"Question {i}: {result.query}")
            text_parts.append(f"Answer: {result.answer}")
            text_parts.append(f"Confidence: {result.confidence_score:.3f}")
            text_parts.append(f"Sources: {', '.join(result.sources) if result.sources else 'None'}")
            text_parts.append(f"Processing Time: {result.processing_time:.2f}s")
            text_parts.append("-" * 50)
        
        return "\n".join(text_parts)


# Example usage and testing
if __name__ == "__main__":
    # Initialize QA Engine
    config = QAEngineConfig(
        # CHANGE THE PATH OF YOUR "vector_db" folder INSIDE local folder
        vector_store_path="C:/local_RAG_bot/local/vector_db",
        collection_name="documents",
        enable_logging=True
    )
    
    qa_engine = QAEngine(config)
    
    # Test questions
    test_questions = [
        "What is python classes?",
        "How to import a python package?",
        "Explain inheritance in python?"
    ]
    
    print("🧪 Testing QA Engine")
    print("=" * 50)
    
    # Get system status
    status = qa_engine.get_system_status()
    print(f"\n System Status:")
    print(f"   Engine Initialized: {status['engine_initialized']}")
    print(f"   Components: {status['components']}")
    
    # Test single question
    print(f"\n🔍 Testing single question...")
    result = qa_engine.ask_question(test_questions[0])
    
    print(f"\n Query: {result.query}")
    print(f" Answer: {result.answer}")
    print(f" Confidence: {result.confidence_score:.3f}")
    print(f"⏱  Processing Time: {result.processing_time:.2f}s")
    
    # Validate answer quality
    is_valid, issues = qa_engine.validate_answer_quality(result)
    print(f"\n Answer Quality: {is_valid}")
    if issues:
        print(f"   Issues: {', '.join(issues)}")
    
    # Test batch processing
    print(f"\n Testing batch processing...")
    batch_results = qa_engine.batch_ask_questions(test_questions)
    
    # Get performance metrics
    metrics = qa_engine.get_performance_metrics(batch_results)
    print(f"\n Performance Metrics:")
    print(f"   Total Questions: {metrics['total_questions']}")
    print(f"   Average Processing Time: {metrics['average_processing_time']:.2f}s")
    print(f"   Average Confidence: {metrics['average_confidence']:.3f}")
    print(f"   Successful Answers: {metrics['successful_answers']}/{metrics['total_questions']}") 