import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

from rag.retrieval.retrieval_engine import RetrievalResult


@dataclass
class ContextSection:
    """Data class for a context section."""
    content: str
    source: str
    relevance_score: float
    section_type: str  # 'main', 'supporting', 'background'


class ContextBuilder:
    """
    Builds coherent context from retrieved document chunks.
    
    Features:
    - Intelligent content combination
    - Redundancy removal
    - Context structuring
    - Relevance-based ordering
    """
    
    def __init__(self, max_context_length: int = 3000):
        """
        Initialize the context builder.
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(__name__)
        
        # Context organization parameters
        self.min_section_length = 50
        self.max_sections = 5
        self.relevance_threshold = 0.4
        
    def build_context(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """
        Build coherent context from retrieval results.
        
        Args:
            query: Original user query
            retrieval_results: List of retrieved document chunks
            
        Returns:
            Structured context string
        """
        if not retrieval_results:
            return "No relevant information found."
        
        # Organize results into sections
        sections = self._organize_into_sections(query, retrieval_results)
        
        # Build structured context
        context = self._create_structured_context(sections)
        
        # Truncate if needed
        if len(context) > self.max_context_length:
            context = self._truncate_context(context)
        
        return context
    
    def _organize_into_sections(self, query: str, 
                               results: List[RetrievalResult]) -> List[ContextSection]:
        """
        Organize retrieval results into logical sections.
        
        Args:
            query: Original query
            results: Retrieval results
            
        Returns:
            List of context sections
        """
        # Group by source and relevance
        source_groups = defaultdict(list)
        
        for result in results:
            source_groups[result.file_name].append(result)
        
        sections = []
        
        # Create main section (highest relevance)
        main_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        if main_results:
            main_content = self._combine_chunks(main_results[:2])  # Top 2 results
            main_section = ContextSection(
                content=main_content,
                source="Multiple sources",
                relevance_score=main_results[0].similarity_score,
                section_type="main"
            )
            sections.append(main_section)
        
        # Create supporting sections
        for source, source_results in source_groups.items():
            if len(sections) >= self.max_sections:
                break
                
            # Skip if already included in main section
            if any(source in section.source for section in sections):
                continue
            
            # Create supporting section
            supporting_content = self._combine_chunks(source_results[:1])
            if len(supporting_content) >= self.min_section_length:
                supporting_section = ContextSection(
                    content=supporting_content,
                    source=source,
                    relevance_score=source_results[0].similarity_score,
                    section_type="supporting"
                )
                sections.append(supporting_section)
        
        return sections
    
    def _combine_chunks(self, chunks: List[RetrievalResult]) -> str:
        """
        Combine multiple chunks into coherent content.
        
        Args:
            chunks: List of retrieval results
            
        Returns:
            Combined content string
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0].content
        
        # Combine chunks with overlap detection
        combined_content = []
        seen_content = set()
        
        for chunk in chunks:
            content = chunk.content.strip()
            
            # Check for overlap with existing content
            is_overlap = False
            for existing in combined_content:
                if self._calculate_overlap(content, existing) > 0.3:  # 30% overlap threshold
                    is_overlap = True
                    break
            
            if not is_overlap and content not in seen_content:
                combined_content.append(content)
                seen_content.add(content)
        
        return "\n\n".join(combined_content)
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate overlap between two text segments.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Overlap ratio between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _create_structured_context(self, sections: List[ContextSection]) -> str:
        """
        Create structured context from sections.
        
        Args:
            sections: List of context sections
            
        Returns:
            Structured context string
        """
        if not sections:
            return "No relevant information available."
        
        context_parts = []
        
        # Add main section
        main_sections = [s for s in sections if s.section_type == "main"]
        if main_sections:
            context_parts.append("## Main Information")
            for section in main_sections:
                context_parts.append(section.content)
        
        # Add supporting sections
        supporting_sections = [s for s in sections if s.section_type == "supporting"]
        if supporting_sections:
            context_parts.append("\n## Additional Information")
            for section in supporting_sections:
                source_info = f"[Source: {section.source}]"
                context_parts.append(f"{source_info}\n{section.content}")
        
        return "\n\n".join(context_parts)
    
    def _truncate_context(self, context: str) -> str:
        """
        Truncate context while preserving structure.
        
        Args:
            context: Full context string
            
        Returns:
            Truncated context string
        """
        if len(context) <= self.max_context_length:
            return context
        
        # Try to truncate at sentence boundaries
        sentences = re.split(r'[.!?]+', context)
        truncated_parts = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + "."
            if current_length + len(sentence) <= self.max_context_length:
                truncated_parts.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        truncated_context = " ".join(truncated_parts)
        
        # Add ellipsis if truncated
        if len(truncated_context) < len(context):
            truncated_context += "..."
        
        return truncated_context
    
    def add_query_context(self, query: str, context: str) -> str:
        """
        Add query information to the context.
        
        Args:
            query: Original user query
            context: Built context
            
        Returns:
            Context with query information
        """
        query_info = f"## User Question\n{query}\n\n"
        return query_info + context
    
    def get_context_statistics(self, context: str) -> Dict:
        """
        Get statistics about the built context.
        
        Args:
            context: Built context string
            
        Returns:
            Dictionary with context statistics
        """
        words = context.split()
        sentences = re.split(r'[.!?]+', context)
        
        return {
            'total_length': len(context),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
            'section_count': context.count('##'),
            'source_count': context.count('[Source:')
        }
    
    def validate_context(self, context: str) -> Tuple[bool, List[str]]:
        """
        Validate the built context for quality.
        
        Args:
            context: Built context string
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum length
        if len(context) < 100:
            issues.append("Context is too short")
        
        # Check for repetitive content
        sentences = re.split(r'[.!?]+', context)
        unique_sentences = set()
        for sentence in sentences:
            clean_sentence = sentence.strip().lower()
            if clean_sentence and len(clean_sentence) > 10:
                if clean_sentence in unique_sentences:
                    issues.append("Contains repetitive content")
                    break
                unique_sentences.add(clean_sentence)
        
        # Check for proper structure
        if not context.count('##'):
            issues.append("Missing section structure")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def enhance_context_with_metadata(self, context: str, 
                                    results: List[RetrievalResult]) -> str:
        """
        Enhance context with metadata information.
        
        Args:
            context: Built context
            results: Retrieval results
            
        Returns:
            Enhanced context with metadata
        """
        if not results:
            return context
        
        # Add source information
        sources = set(result.file_name for result in results)
        source_info = f"\n## Sources\n"
        source_info += "\n".join([f"- {source}" for source in sources])
        
        # Add confidence information
        avg_confidence = sum(result.similarity_score for result in results) / len(results)
        confidence_info = f"\n## Confidence\nAverage relevance score: {avg_confidence:.3f}"
        
        return context + source_info + confidence_info


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize context builder
    builder = ContextBuilder()
    
    # Mock retrieval results for testing
    from dataclasses import dataclass
    
    @dataclass
    class MockRetrievalResult:
        content: str
        file_name: str
        similarity_score: float
    
    mock_results = [
        MockRetrievalResult(
            content="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
            file_name="ai_overview.txt",
            similarity_score=0.85
        ),
        MockRetrievalResult(
            content="Machine Learning is a subset of AI that enables computers to learn from experience.",
            file_name="ml_intro.txt",
            similarity_score=0.78
        ),
        MockRetrievalResult(
            content="Deep Learning uses neural networks with multiple layers to understand complex patterns.",
            file_name="deep_learning.txt",
            similarity_score=0.72
        )
    ]
    
    # Test context building
    query = "What is python classes?"
    
    print(" Testing Context Builder")
    print("=" * 50)
    
    # Build context
    context = builder.build_context(query, mock_results)
    
    print(f"\n Query: {query}")
    print(f"\n Built Context:")
    print("-" * 30)
    print(context)
    
    # Get statistics
    stats = builder.get_context_statistics(context)
    print(f"\n📊 Context Statistics:")
    print(f"   Length: {stats['total_length']} characters")
    print(f"   Words: {stats['word_count']}")
    print(f"   Sentences: {stats['sentence_count']}")
    
    # Validate context
    is_valid, issues = builder.validate_context(context)
    print(f"\n Context Valid: {is_valid}")
    if issues:
        print(f"   Issues: {', '.join(issues)}") 