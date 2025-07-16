import re
import string
from typing import Dict, List, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class QueryProcessor:
    """
    Processes and understands user queries for the QA system.
    
    Features:
    - Query normalization and cleaning
    - Keyword extraction
    - Intent recognition
    - Query expansion
    """
    
    def __init__(self):
        """Initialize the query processor with NLP tools."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Question words for intent recognition
        self.question_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose'
        }
        
        # Common question patterns
        self.question_patterns = {
            'definition': r'\b(what is|what are|define|definition of)\b',
            'comparison': r'\b(compare|difference between|similarities|versus|vs)\b',
            'process': r'\b(how to|how does|steps to|process of)\b',
            'cause_effect': r'\b(why|because|reason|cause|effect)\b',
            'example': r'\b(example|instance|case|illustration)\b'
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize and clean the user query.
        
        Args:
            query: Raw user query
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove punctuation except for question marks
        query = re.sub(r'[^\w\s\?]', '', query)
        
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query.
        
        Args:
            query: Normalized query string
            
        Returns:
            List of extracted keywords
        """
        # Tokenize the query
        tokens = word_tokenize(query)
        
        # Remove stop words and lemmatize
        keywords = []
        for token in tokens:
            if token.lower() not in self.stop_words and len(token) > 2:
                # Lemmatize the token
                lemma = self.lemmatizer.lemmatize(token.lower())
                keywords.append(lemma)
        
        return keywords
    
    def recognize_intent(self, query: str) -> Dict[str, any]:
        """
        Recognize the intent and type of the question.
        
        Args:
            query: Normalized query string
            
        Returns:
            Dictionary with intent information
        """
        intent_info = {
            'question_type': 'general',
            'question_word': None,
            'intent_category': 'information',
            'confidence': 0.5
        }
        
        # Check for question words
        words = query.lower().split()
        for word in words:
            if word in self.question_words:
                intent_info['question_word'] = word
                break
        
        # Check for specific patterns
        for pattern_name, pattern in self.question_patterns.items():
            if re.search(pattern, query.lower()):
                intent_info['intent_category'] = pattern_name
                intent_info['confidence'] = 0.8
                break
        
        # Determine question type based on question word
        if intent_info['question_word']:
            if intent_info['question_word'] in ['what', 'which']:
                intent_info['question_type'] = 'definition'
            elif intent_info['question_word'] in ['how']:
                intent_info['question_type'] = 'process'
            elif intent_info['question_word'] in ['why']:
                intent_info['question_type'] = 'cause_effect'
            elif intent_info['question_word'] in ['when', 'where']:
                intent_info['question_type'] = 'factual'
            elif intent_info['question_word'] in ['who']:
                intent_info['question_type'] = 'person'
        
        return intent_info
    
    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Create query variations for better retrieval.
        
        Args:
            query: Original query
            keywords: Extracted keywords
            
        Returns:
            List of expanded query variations
        """
        variations = [query]
        
        # Add keyword-based variations
        if keywords:
            keyword_query = ' '.join(keywords[:3])  # Use top 3 keywords
            if keyword_query != query:
                variations.append(keyword_query)
        
        # In a full implementation, you'd use a thesaurus or word embeddings
        synonym_map = {
            'ai': ['artificial intelligence', 'machine learning'],
            'ml': ['machine learning', 'artificial intelligence'],
            'nlp': ['natural language processing', 'text processing'],
            'deep learning': ['neural networks', 'ai'],
            'algorithm': ['method', 'technique', 'approach']
        }
        
        for keyword in keywords:
            if keyword in synonym_map:
                for synonym in synonym_map[keyword]:
                    variation = query.replace(keyword, synonym)
                    if variation not in variations:
                        variations.append(variation)
        
        return variations
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Complete query processing pipeline.
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary with processed query information
        """
        # Normalize the query
        normalized_query = self.normalize_query(query)
        
        # Extract keywords
        keywords = self.extract_keywords(normalized_query)
        
        # Recognize intent
        intent_info = self.recognize_intent(normalized_query)
        
        # Expand query
        query_variations = self.expand_query(normalized_query, keywords)
        
        return {
            'original_query': query,
            'normalized_query': normalized_query,
            'keywords': keywords,
            'intent': intent_info,
            'query_variations': query_variations,
            'processed_at': '2024-01-01T00:00:00Z'  # In real implementation, use datetime
        }
    
    def get_query_statistics(self, query: str) -> Dict[str, any]:
        """
        Get statistics about the processed query.
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary with query statistics
        """
        processed = self.process_query(query)
        
        return {
            'query_length': len(query),
            'word_count': len(query.split()),
            'keyword_count': len(processed['keywords']),
            'intent_confidence': processed['intent']['confidence'],
            'variations_count': len(processed['query_variations'])
        }


# Example usage and testing
if __name__ == "__main__":
    processor = QueryProcessor()
    
    # Test queries
    #TEST WITH YOUR CUSTOM QUERIES BASED ON YOUR DOCUMENTS IN data FOLDER
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Why is deep learning important?",
        "Compare AI and traditional programming",
        "Give me an example of NLP applications"
    ]
    
    print("Testing Query Processor")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = processor.process_query(query)
        
        print(f"   Keywords: {result['keywords']}")
        print(f"   Intent: {result['intent']['question_type']} ({result['intent']['intent_category']})")
        print(f"   Confidence: {result['intent']['confidence']}")
        print(f"   Variations: {len(result['query_variations'])}") 