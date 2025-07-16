import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import requests
import time

from rag.retrieval.query_processor import QueryProcessor


@dataclass
class GeneratedAnswer:
    """Data class for generated answers."""
    answer: str
    confidence_score: float
    sources_used: List[str]
    reasoning: str
    answer_type: str  # 'direct', 'analytical', 'comparative', 'explanatory'


class AnswerGenerator:
    """
    Generates intelligent answers using Ollama.
    
    Features:
    - Context-aware answer generation
    - Prompt engineering
    - Answer quality assessment
    - Multiple answer types
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", 
                 model_name: str = "mistral:7b"):
        """
        Initialize the answer generator.
        
        Args:
            ollama_url: Ollama server URL
            model_name: Name of the Ollama model to use
        """
        # Hardcode the endpoint and model name
        self.ollama_url = "http://localhost:11434"
        self.model_name = "mistral:7b"
        self.logger = logging.getLogger(__name__)
        self.query_processor = QueryProcessor()
        
        # Answer generation parameters
        self.max_answer_length = 1000
        self.min_confidence_threshold = 0.3
        self.max_retries = 3
        
        # Prompt templates
        self.prompt_templates = {
            'general': self._get_general_prompt(),
            'definition': self._get_definition_prompt(),
            'comparison': self._get_comparison_prompt(),
            'process': self._get_process_prompt(),
            'analytical': self._get_analytical_prompt()
        }
    
    def _get_general_prompt(self) -> str:
        """Get the general prompt template."""
        return """You are a helpful AI assistant. Based on the provided context, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based only on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise but comprehensive
4. Use clear, simple language
5. Include relevant details from the context

Answer:"""
    
    def _get_definition_prompt(self) -> str:
        """Get the definition prompt template."""
        return """You are an AI assistant specializing in providing clear definitions. Based on the context, provide a comprehensive definition.

Context:
{context}

Question: {query}

Instructions:
1. Provide a clear, accurate definition
2. Include key characteristics and features
3. Use simple, understandable language
4. If applicable, mention related concepts
5. Be comprehensive but concise

Definition:"""
    
    def _get_comparison_prompt(self) -> str:
        """Get the comparison prompt template."""
        return """You are an AI assistant that helps with comparisons. Based on the context, provide a detailed comparison.

Context:
{context}

Question: {query}

Instructions:
1. Identify the items to compare
2. List similarities and differences
3. Provide specific examples from the context
4. Use clear, structured format
5. Be objective and balanced

Comparison:"""
    
    def _get_process_prompt(self) -> str:
        """Get the process prompt template."""
        return """You are an AI assistant that explains processes and procedures. Based on the context, explain the process clearly.

Context:
{context}

Question: {query}

Instructions:
1. Break down the process into clear steps
2. Explain each step in detail
3. Use logical sequence
4. Include important details from context
5. Make it easy to follow

Process Explanation:"""
    
    def _get_analytical_prompt(self) -> str:
        """Get the analytical prompt template."""
        return """You are an AI assistant that provides analytical insights. Based on the context, provide a thoughtful analysis.

Context:
{context}

Question: {query}

Instructions:
1. Analyze the topic thoroughly
2. Consider multiple perspectives
3. Provide evidence from the context
4. Draw logical conclusions
5. Be insightful and well-reasoned

Analysis:"""
    
    def generate_answer(self, query: str, context: str) -> GeneratedAnswer:
        """
        Generate an answer using Ollama.
        
        Args:
            query: User's question
            context: Built context from documents
            
        Returns:
            Generated answer with metadata
        """
        try:
            # Process query to determine answer type
            processed_query = self.query_processor.process_query(query)
            answer_type = self._determine_answer_type(processed_query)
            
            # Select appropriate prompt template
            prompt_template = self.prompt_templates.get(answer_type, self.prompt_templates['general'])
            
            # Build the prompt
            prompt = prompt_template.format(
                context=context,
                query=query
            )
            
            # Generate answer using Ollama
            raw_answer = self._call_ollama(prompt)
            
            # Process and structure the answer
            processed_answer = self._process_answer(raw_answer, query, context)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(processed_answer, context, query)
            
            # Extract sources from context
            sources = self._extract_sources_from_context(context)
            
            return GeneratedAnswer(
                answer=processed_answer,
                confidence_score=confidence,
                sources_used=sources,
                reasoning=self._generate_reasoning(query, processed_answer),
                answer_type=answer_type
            )
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return self._generate_fallback_answer(query, context)
    
    def _determine_answer_type(self, processed_query: Dict) -> str:
        """
        Determine the type of answer to generate.
        
        Args:
            processed_query: Processed query information
            
        Returns:
            Answer type string
        """
        intent = processed_query['intent']
        question_type = intent.get('question_type', 'general')
        intent_category = intent.get('intent_category', 'information')
        
        # Map question types to answer types
        type_mapping = {
            'definition': 'definition',
            'process': 'process',
            'cause_effect': 'analytical',
            'factual': 'general',
            'person': 'general',
            'general': 'general'
        }
        
        # Override based on intent category
        if intent_category == 'comparison':
            return 'comparison'
        elif intent_category == 'analytical':
            return 'analytical'
        
        return type_mapping.get(question_type, 'general')
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate response.
        
        Args:
            prompt: Complete prompt for Ollama
            
        Returns:
            Generated response from Ollama
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    self.logger.warning(f"Ollama API error (attempt {attempt + 1}): {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"Ollama call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Wait before retry
        
        raise Exception("Failed to generate answer after all retries")
    
    def _process_answer(self, raw_answer: str, query: str, context: str) -> str:
        """
        Process and clean the raw answer from Ollama.
        
        Args:
            raw_answer: Raw answer from Ollama
            query: Original query
            context: Used context
            
        Returns:
            Processed answer
        """
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Answer:", "Response:", "Based on the context:", "According to the context:"
        ]
        
        processed_answer = raw_answer
        for prefix in prefixes_to_remove:
            if processed_answer.startswith(prefix):
                processed_answer = processed_answer[len(prefix):].strip()
        
        # Truncate if too long
        if len(processed_answer) > self.max_answer_length:
            sentences = re.split(r'[.!?]+', processed_answer)
            truncated_parts = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + "."
                if current_length + len(sentence) <= self.max_answer_length:
                    truncated_parts.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            processed_answer = " ".join(truncated_parts)
            if len(processed_answer) < len(raw_answer):
                processed_answer += "..."
        
        return processed_answer
    
    def _calculate_confidence(self, answer: str, context: str, query: str) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            answer: Generated answer
            context: Used context
            query: Original query
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Check if answer addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
        confidence += query_coverage * 0.2
        
        # Check answer length (prefer substantial answers)
        if 50 <= len(answer) <= 500:
            confidence += 0.1
        
        # Check if answer uses context information
        context_words = set(context.lower().split())
        context_usage = len(answer_words.intersection(context_words)) / max(len(answer_words), 1)
        confidence += context_usage * 0.2
        
        # Check for specific answer patterns
        if any(word in answer.lower() for word in ['because', 'since', 'therefore', 'thus']):
            confidence += 0.1  # Reasoning indicators
        
        return min(confidence, 1.0)
    
    def _extract_sources_from_context(self, context: str) -> List[str]:
        """
        Extract source information from context.
        
        Args:
            context: Built context
            
        Returns:
            List of source file names
        """
        sources = []
        
        # Look for source patterns in context
        source_patterns = [
            r'\[Source: ([^\]]+)\]',
            r'From: ([^\n]+)',
            r'Source: ([^\n]+)'
        ]
        
        for pattern in source_patterns:
            matches = re.findall(pattern, context)
            sources.extend(matches)
        
        return list(set(sources))  # Remove duplicates
    
    def _generate_reasoning(self, query: str, answer: str) -> str:
        """
        Generate reasoning for the answer.
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            Reasoning string
        """
        reasoning_parts = []
        
        # Check if answer directly addresses the query
        query_keywords = set(query.lower().split())
        answer_keywords = set(answer.lower().split())
        keyword_match = len(query_keywords.intersection(answer_keywords))
        
        if keyword_match > 0:
            reasoning_parts.append(f"Answer addresses {keyword_match} key terms from the query")
        
        # Check answer structure
        if len(answer) > 100:
            reasoning_parts.append("Answer provides substantial detail")
        elif len(answer) > 50:
            reasoning_parts.append("Answer provides adequate detail")
        else:
            reasoning_parts.append("Answer is concise")
        
        # Check for reasoning indicators
        if any(word in answer.lower() for word in ['because', 'since', 'therefore']):
            reasoning_parts.append("Answer includes logical reasoning")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Answer generated based on available context"
    
    def _generate_fallback_answer(self, query: str, context: str) -> GeneratedAnswer:
        """
        Generate a fallback answer when Ollama fails.
        
        Args:
            query: Original query
            context: Available context
            
        Returns:
            Fallback answer
        """
        fallback_answer = f"I apologize, but I'm unable to generate a complete answer at the moment. "
        fallback_answer += f"However, I found some relevant information in the documents that might help answer your question: '{query}'"
        
        if context and context != "No relevant information found.":
            # Extract a brief summary from context
            sentences = re.split(r'[.!?]+', context)
            if sentences:
                fallback_answer += f" The documents mention: {sentences[0][:100]}..."
        
        return GeneratedAnswer(
            answer=fallback_answer,
            confidence_score=0.1,
            sources_used=[],
            reasoning="Fallback answer due to generation failure",
            answer_type="fallback"
        )
    
    def validate_answer(self, answer: GeneratedAnswer) -> Tuple[bool, List[str]]:
        """
        Validate the generated answer for quality.
        
        Args:
            answer: Generated answer
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum length
        if len(answer.answer) < 20:
            issues.append("Answer is too short")
        
        # Check confidence threshold
        if answer.confidence_score < self.min_confidence_threshold:
            issues.append("Answer confidence is too low")
        
        # Check for generic responses
        generic_phrases = [
            "I don't have enough information",
            "I cannot answer",
            "I'm not sure",
            "I don't know"
        ]
        
        if any(phrase in answer.answer.lower() for phrase in generic_phrases):
            issues.append("Answer is too generic")
        
        # Check for repetition
        words = answer.answer.lower().split()
        if len(set(words)) / len(words) < 0.7:  # Less than 70% unique words
            issues.append("Answer contains repetitive content")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize answer generator
    generator = AnswerGenerator()
    
    # Test context and query
    #TEST WITH YOUR CUSTOM CONTEXT AND QUERY BASED ON YOUR DOCUMENTS IN data FOLDER
    test_context = """
    ## Main Information
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
    These machines can perform tasks that typically require human intelligence, such as visual perception,
    speech recognition, decision-making, and language translation.
    
    Machine Learning is a subset of AI that enables computers to learn and improve from experience
    without being explicitly programmed. It uses algorithms to identify patterns in data and make
    predictions or decisions.
    
    ## Additional Information
    [Source: ai_overview.txt]
    Deep Learning is a type of machine learning that uses neural networks with multiple layers
    to model and understand complex patterns in data.
    """
    
    test_query = "What is artificial intelligence?"
    
    print(" Testing Answer Generator")
    print("=" * 50)
    
    try:
        # Generate answer
        answer = generator.generate_answer(test_query, test_context)
        
        print(f"\n Query: {test_query}")
        print(f"\n Generated Answer:")
        print("-" * 30)
        print(answer.answer)
        
        print(f"\n Answer Statistics:")
        print(f"   Confidence: {answer.confidence_score:.3f}")
        print(f"   Type: {answer.answer_type}")
        print(f"   Sources: {answer.sources_used}")
        print(f"   Reasoning: {answer.reasoning}")
        
        # Validate answer
        is_valid, issues = generator.validate_answer(answer)
        print(f"\n Answer Valid: {is_valid}")
        if issues:
            print(f"   Issues: {', '.join(issues)}")
            
    except Exception as e:
        print(f" Error during testing: {e}")
        print("This might be due to Ollama not running or model not available.") 