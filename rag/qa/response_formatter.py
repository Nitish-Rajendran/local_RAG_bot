import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re

from rag.qa.answer_generator import GeneratedAnswer


@dataclass
class FormattedResponse:
    """Data class for formatted responses."""
    answer: str
    confidence_score: float
    sources: List[str]
    metadata: Dict
    formatted_text: str
    json_response: str


class ResponseFormatter:
    """
    Formats and structures generated answers for presentation.
    
    Features:
    - Answer structuring and formatting
    - Source citation formatting
    - Confidence indicators
    - Metadata inclusion
    - Multiple output formats
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.max_answer_length = 1000
        self.include_metadata = True
        self.include_sources = True
        self.include_confidence = True
        
    def format_response(self, answer: GeneratedAnswer, 
                       query: str, 
                       processing_time: float = None) -> FormattedResponse:
        """
        Format a generated answer into a structured response.
        
        Args:
            answer: Generated answer
            query: Original user query
            processing_time: Time taken to generate answer
            
        Returns:
            Formatted response object
        """
        # Build formatted text
        formatted_text = self._build_formatted_text(answer, query, processing_time)
        
        # Create metadata
        metadata = self._create_metadata(answer, query, processing_time)
        
        # Create JSON response
        json_response = self._create_json_response(answer, query, metadata)
        
        return FormattedResponse(
            answer=answer.answer,
            confidence_score=answer.confidence_score,
            sources=answer.sources_used,
            metadata=metadata,
            formatted_text=formatted_text,
            json_response=json_response
        )
    
    def _build_formatted_text(self, answer: GeneratedAnswer, 
                             query: str, processing_time: float) -> str:
        """
        Build formatted text response.
        
        Args:
            answer: Generated answer
            query: Original query
            processing_time: Processing time
            
        Returns:
            Formatted text string
        """
        parts = []
        
        # Add answer
        parts.append("🤖 ANSWER")
        parts.append("=" * 50)
        parts.append(answer.answer)
        parts.append("")
        
        # Add confidence indicator
        if self.include_confidence:
            confidence_level = self._get_confidence_level(answer.confidence_score)
            parts.append(f" Confidence: {confidence_level} ({answer.confidence_score:.1%})")
            parts.append("")
        
        # Add sources
        if self.include_sources and answer.sources_used:
            parts.append(" Sources:")
            for source in answer.sources_used:
                parts.append(f"   • {source}")
            parts.append("")
        
        # Add reasoning
        if answer.reasoning:
            parts.append(" Reasoning:")
            parts.append(f"   {answer.reasoning}")
            parts.append("")
        
        # Add metadata
        if self.include_metadata:
            parts.append("ℹ  Metadata:")
            parts.append(f"   • Answer Type: {answer.answer_type}")
            parts.append(f"   • Processing Time: {processing_time:.2f}s" if processing_time else "   • Processing Time: N/A")
            parts.append(f"   • Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(parts)
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Get human-readable confidence level.
        
        Args:
            confidence_score: Confidence score between 0 and 1
            
        Returns:
            Confidence level string
        """
        if confidence_score >= 0.8:
            return "Very High"
        elif confidence_score >= 0.6:
            return "High"
        elif confidence_score >= 0.4:
            return "Medium"
        elif confidence_score >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _create_metadata(self, answer: GeneratedAnswer, 
                        query: str, processing_time: float) -> Dict:
        """
        Create metadata for the response.
        
        Args:
            answer: Generated answer
            query: Original query
            processing_time: Processing time
            
        Returns:
            Metadata dictionary
        """
        return {
            "query": query,
            "answer_type": answer.answer_type,
            "confidence_score": answer.confidence_score,
            "confidence_level": self._get_confidence_level(answer.confidence_score),
            "sources_count": len(answer.sources_used),
            "answer_length": len(answer.answer),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "reasoning": answer.reasoning
        }
    
    def _create_json_response(self, answer: GeneratedAnswer, 
                            query: str, metadata: Dict) -> str:
        """
        Create JSON response.
        
        Args:
            answer: Generated answer
            query: Original query
            metadata: Response metadata
            
        Returns:
            JSON response string
        """
        response_data = {
            "query": query,
            "answer": answer.answer,
            "confidence": {
                "score": answer.confidence_score,
                "level": self._get_confidence_level(answer.confidence_score)
            },
            "sources": answer.sources_used,
            "metadata": metadata
        }
        
        return json.dumps(response_data, indent=2)
    
    def format_simple_response(self, answer: GeneratedAnswer) -> str:
        """
        Format a simple text response without metadata.
        
        Args:
            answer: Generated answer
            
        Returns:
            Simple formatted response
        """
        parts = []
        
        # Add answer
        parts.append(answer.answer)
        
        # Add confidence if low
        if answer.confidence_score < 0.5:
            parts.append(f"\n[Confidence: {answer.confidence_score:.1%}]")
        
        # Add sources if available
        if answer.sources_used:
            parts.append(f"\nSources: {', '.join(answer.sources_used)}")
        
        return "\n".join(parts)
    
    def format_markdown_response(self, answer: GeneratedAnswer, 
                               query: str) -> str:
        """
        Format response as Markdown.
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            Markdown formatted response
        """
        parts = []
        
        # Add query
        parts.append(f"## Query\n{query}\n")
        
        # Add answer
        parts.append("## Answer")
        parts.append(answer.answer)
        parts.append("")
        
        # Add confidence
        confidence_level = self._get_confidence_level(answer.confidence_score)
        parts.append(f"**Confidence:** {confidence_level} ({answer.confidence_score:.1%})")
        parts.append("")
        
        # Add sources
        if answer.sources_used:
            parts.append("**Sources:**")
            for source in answer.sources_used:
                parts.append(f"- {source}")
            parts.append("")
        
        # Add reasoning
        if answer.reasoning:
            parts.append("**Reasoning:**")
            parts.append(answer.reasoning)
        
        return "\n".join(parts)
    
    def format_html_response(self, answer: GeneratedAnswer, 
                           query: str) -> str:
        """
        Format response as HTML.
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            HTML formatted response
        """
        html_parts = []
        
        # Add query
        html_parts.append(f"<h3>Query</h3>")
        html_parts.append(f"<p>{query}</p>")
        
        # Add answer
        html_parts.append(f"<h3>Answer</h3>")
        html_parts.append(f"<p>{answer.answer}</p>")
        
        # Add confidence
        confidence_level = self._get_confidence_level(answer.confidence_score)
        html_parts.append(f"<p><strong>Confidence:</strong> {confidence_level} ({answer.confidence_score:.1%})</p>")
        
        # Add sources
        if answer.sources_used:
            html_parts.append(f"<h4>Sources</h4>")
            html_parts.append("<ul>")
            for source in answer.sources_used:
                html_parts.append(f"<li>{source}</li>")
            html_parts.append("</ul>")
        
        # Add reasoning
        if answer.reasoning:
            html_parts.append(f"<h4>Reasoning</h4>")
            html_parts.append(f"<p>{answer.reasoning}</p>")
        
        return "\n".join(html_parts)
    
    def validate_formatted_response(self, response: FormattedResponse) -> tuple[bool, list[str]]:
        """
        Validate the formatted response.
        
        Args:
            response: Formatted response
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if answer is present
        if not response.answer or len(response.answer.strip()) == 0:
            issues.append("Answer is empty")
        
        # Check answer length
        if len(response.answer) > self.max_answer_length:
            issues.append("Answer is too long")
        
        # Check confidence score
        if not (0 <= response.confidence_score <= 1):
            issues.append("Invalid confidence score")
        
        # Check JSON response
        try:
            json.loads(response.json_response)
        except json.JSONDecodeError:
            issues.append("Invalid JSON response")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_response_statistics(self, response: FormattedResponse) -> Dict:
        """
        Get statistics about the formatted response.
        
        Args:
            response: Formatted response
            
        Returns:
            Dictionary with response statistics
        """
        return {
            "answer_length": len(response.answer),
            "formatted_length": len(response.formatted_text),
            "json_length": len(response.json_response),
            "sources_count": len(response.sources),
            "confidence_score": response.confidence_score,
            "has_reasoning": bool(response.metadata.get("reasoning")),
            "processing_time": response.metadata.get("processing_time")
        }


# Example usage and testing
if __name__ == "__main__":
    # Mock generated answer for testing
    from dataclasses import dataclass
    
    @dataclass
    class MockGeneratedAnswer:
        answer: str
        confidence_score: float
        sources_used: List[str]
        reasoning: str
        answer_type: str
    #TEST WITH YOUR CUSTOM MOCK ANSWER AND QUERY BASED ON YOUR DOCUMENTS IN data FOLDER
    mock_answer = MockGeneratedAnswer(
        answer="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks typically requiring human intelligence.",
        confidence_score=0.85,
        sources_used=["ai_overview.txt", "ml_intro.txt"],
        reasoning="Answer directly addresses the query about AI definition and includes key characteristics",
        answer_type="definition"
    )
    
    # Initialize formatter
    formatter = ResponseFormatter()
    
    # Test formatting
    query = "What is artificial intelligence?"
    processing_time = 2.5
    
    print("🧪 Testing Response Formatter")
    print("=" * 50)
    
    # Format response
    formatted_response = formatter.format_response(mock_answer, query, processing_time)
    
    print("\n Formatted Response:")
    print("-" * 30)
    print(formatted_response.formatted_text)
    
    # Test different formats
    print("\n Simple Format:")
    print("-" * 30)
    print(formatter.format_simple_response(mock_answer))
    
    print("\n Markdown Format:")
    print("-" * 30)
    print(formatter.format_markdown_response(mock_answer, query))
    
    # Get statistics
    stats = formatter.get_response_statistics(formatted_response)
    print(f"\n Response Statistics:")
    print(f"   Answer Length: {stats['answer_length']} characters")
    print(f"   Sources: {stats['sources_count']}")
    print(f"   Confidence: {stats['confidence_score']:.1%}")
    
    # Validate response
    is_valid, issues = formatter.validate_formatted_response(formatted_response)
    print(f"\n Response Valid: {is_valid}")
    if issues:
        print(f"   Issues: {', '.join(issues)}") 