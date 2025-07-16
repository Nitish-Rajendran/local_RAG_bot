import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{package} installed successfully")
            return True
        else:
            print(f"Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error installing {package}: {e}")
        return False

def install_core_packages():
    """Install core packages one by one"""
    print("Installing core packages...")
    
    core_packages = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.22",
        "pypdf>=3.17.4"
    ]
    
    failed_packages = []
    
    for package in core_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    return failed_packages

def install_ai_packages():
    """Install AI/ML packages"""
    print("\n🤖 Installing AI/ML packages...")
    
    ai_packages = [
        "torch>=2.2.0",
        "transformers>=4.36.2",
        "sentence-transformers>=2.2.2"
    ]
    
    failed_packages = []
    
    for package in ai_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    return failed_packages

def install_interface_packages():
    """Install web interface packages"""
    print("\nInstalling interface packages...")
    
    interface_packages = [
        "gradio>=4.15.0",
        "streamlit>=1.29.0"
    ]
    
    failed_packages = []
    
    for package in interface_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    return failed_packages

def install_utility_packages():
    """Install utility packages"""
    print("\n🔧 Installing utility packages...")
    
    utility_packages = [
        "psutil>=5.9.6",
        "requests>=2.31.0",
        "numpy>=1.24.3",
        "pandas>=2.1.4",
        "python-docx>=1.1.0",
        "python-pptx>=0.6.23",
        "openpyxl>=3.1.2"
    ]
    
    failed_packages = []
    
    for package in utility_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    return failed_packages

def install_optional_packages():
    """Install optional packages"""
    print("\nInstalling optional packages...")
    
    optional_packages = [
        "faiss-cpu>=1.7.4"
    ]
    
    failed_packages = []
    
    for package in optional_packages:
        if not install_package(package):
            print(f"Optional package {package} failed - continuing...")
            failed_packages.append(package)
    
    return failed_packages

def main():
    """Install all packages with better error handling"""
    print("RAG System - Package Installation")
    print("=" * 50)
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install packages in groups
    all_failed = []
    
    # Core packages
    failed = install_core_packages()
    all_failed.extend(failed)
    
    # AI packages
    failed = install_ai_packages()
    all_failed.extend(failed)
    
    # Interface packages
    failed = install_interface_packages()
    all_failed.extend(failed)
    
    # Utility packages
    failed = install_utility_packages()
    all_failed.extend(failed)
    
    # Optional packages
    failed = install_optional_packages()
    all_failed.extend(failed)
    
    # Summary
    print("\n" + "=" * 50)
    print("INSTALLATION SUMMARY:")
    print("=" * 50)
    
    if all_failed:
        print(f"Failed packages ({len(all_failed)}):")
        for package in all_failed:
            print(f"   - {package}")
        print("\nYou can try installing these manually:")
        for package in all_failed:
            print(f"   pip install {package}")
    else:
        print("All packages installed successfully!")

if __name__ == "__main__":
    main() 