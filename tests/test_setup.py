import subprocess
import sys
import os
from pathlib import Path

def test_python_imports():
    """Test that all required Python packages can be imported"""
    print("Testing Python imports...")
    
    packages_to_test = [
        'langchain',
        'chromadb',
        'pypdf',
        'sentence_transformers',
        'gradio',
        'streamlit',
        'torch',
        'transformers'
    ]
    
    failed_imports = []
    
    for package in packages_to_test:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("All Python packages imported successfully!")
        return True

def test_ollama():
    """Test that Ollama is working"""
    print("\nTesting Ollama...")
    
    try:
        # Test ollama version
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f" Ollama version: {result.stdout.strip()}")
        else:
            print("Ollama not responding")
            return False
        
        # Test model availability
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Ollama models:")
            print(result.stdout)
            
            # Check if mistral:7b is available
            if 'mistral:7b' in result.stdout:
                print(" mistral:7b model is available")
                
                # Test a simple query
                print("Testing model response...")
                test_query = "Say hello in one sentence."
                result = subprocess.run(['ollama', 'run', 'mistral:7b', test_query], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and result.stdout.strip():
                    print("Model response test successful!")
                    print(f"Response: {result.stdout.strip()[:100]}...")
                    return True
                else:
                    print("Model response test failed")
                    return False
            else:
                print("mistral:7b model not found")
                return False
        else:
            print("Could not list models")
            return False
            
    except Exception as e:
        print(f"Ollama test error: {e}")
        return False

def test_embeddings():
    """Test sentence transformers embeddings"""
    print("\nTesting embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding generation
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(sentences)
        
        if embeddings.shape[0] == len(sentences):
            print("Embeddings generated successfully!")
            print(f"Embedding shape: {embeddings.shape}")
            return True
        else:
            print("Embedding generation failed")
            return False
            
    except Exception as e:
        print(f"Embedding test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing RAG System")
    print("=" * 50)
    
    tests = [
        ("Python Imports", test_python_imports),
        ("Embeddings", test_embeddings),
        ("Ollama", test_ollama)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Please fix the issues before proceeding.")
        print("Check the error messages above for guidance.")

if __name__ == "__main__":
    main() 