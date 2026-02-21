"""
Notebook Helper for Windows Users
This module provides Windows-compatible alternatives for common notebook operations
"""

import os
import shutil
import requests
from pathlib import Path


def download_file(url, filename=None, destination='.'):
    """
    Windows-compatible file download (replaces wget)
    
    Args:
        url (str): URL to download from
        filename (str): Optional filename. If None, extracted from URL
        destination (str): Directory to save file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if filename is None:
            filename = url.split('/')[-1]
        
        filepath = os.path.join(destination, filename)
        
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
                print()  # New line after progress
            else:
                f.write(response.content)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ Downloaded: {filepath} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False


def load_phishtank_data(use_local=True):
    """
    Load PhishTank data - Windows compatible
    Replaces: !wget http://data.phishtank.com/data/online-valid.csv
    
    Args:
        use_local (bool): Try to use local file first
    
    Returns:
        str: Path to the loaded file
    """
    local_file = 'DataFiles/2.online-valid.csv'
    target_file = 'online-valid.csv'
    
    # Try to use existing local file
    if use_local and os.path.exists(local_file):
        print(f"✓ Found existing data: {local_file}")
        if not os.path.exists(target_file):
            shutil.copy(local_file, target_file)
            print(f"✓ Copied to {target_file}")
        return target_file
    
    # Download fresh data
    url = "http://data.phishtank.com/data/online-valid.csv"
    if download_file(url, target_file):
        return target_file
    
    # Fallback to local if download failed
    if os.path.exists(local_file):
        print(f"Download failed, using local file: {local_file}")
        shutil.copy(local_file, target_file)
        return target_file
    
    print("\n⚠ Could not load PhishTank data")
    print("Please download manually from: https://www.phishtank.com/developer_info.php")
    return None


def load_benign_urls():
    """
    Load benign URLs dataset
    
    Returns:
        str: Path to the benign URLs file
    """
    benign_file = 'DataFiles/1.Benign_list_big_final.csv'
    
    if os.path.exists(benign_file):
        print(f"✓ Found benign URLs: {benign_file}")
        return benign_file
    
    print(f"✗ Benign URL file not found: {benign_file}")
    print("Please download from: https://www.unb.ca/cic/datasets/url-2016.html")
    return None


def check_data_files():
    """
    Check if all required data files exist
    
    Returns:
        dict: Status of each data file
    """
    files = {
        'Benign URLs (large)': 'DataFiles/1.Benign_list_big_final.csv',
        'PhishTank URLs': 'DataFiles/2.online-valid.csv',
        'Legitimate (5000)': 'DataFiles/3.legitimate.csv',
        'Phishing (5000)': 'DataFiles/4.phishing.csv',
        'Combined Dataset': 'DataFiles/5.urldata.csv',
    }
    
    print("=" * 60)
    print("Data Files Status")
    print("=" * 60)
    
    status = {}
    for name, filepath in files.items():
        exists = os.path.exists(filepath)
        status[name] = exists
        
        if exists:
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✓ {name:25s}: {filepath} ({size:.2f} MB)")
        else:
            print(f"✗ {name:25s}: NOT FOUND")
    
    print("=" * 60)
    return status


def setup_notebook_environment():
    """
    Set up the environment for notebooks
    Prints useful information and checks requirements
    """
    import sys
    import pandas as pd
    
    print("=" * 60)
    print("Notebook Environment Setup")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pandas: {pd.__version__}")
    print(f"Working Directory: {os.getcwd()}")
    print()
    
    # Check data files
    status = check_data_files()
    
    print()
    if all(status.values()):
        print("✓ All data files present!")
    else:
        print("⚠ Some data files are missing")
        print("  You can still use the combined dataset (5.urldata.csv)")
    
    print()
    print("Helper functions available:")
    print("  - download_file(url, filename)")
    print("  - load_phishtank_data()")
    print("  - load_benign_urls()")
    print("  - check_data_files()")
    print("=" * 60)


# Example usage function
def get_example_usage():
    """Print example usage of helper functions"""
    examples = """
    
    EXAMPLE USAGE IN NOTEBOOK:
    ==========================
    
    # Import the helper
    import notebook_helper as nh
    
    # Check data files
    nh.check_data_files()
    
    # Load PhishTank data (Windows-compatible, replaces wget)
    phishtank_file = nh.load_phishtank_data()
    
    # Load benign URLs
    benign_file = nh.load_benign_urls()
    
    # Download any file
    nh.download_file('http://example.com/data.csv', 'mydata.csv')
    
    # Setup environment and check everything
    nh.setup_notebook_environment()
    
    """
    print(examples)


if __name__ == "__main__":
    # When run as script, show environment setup
    setup_notebook_environment()
    print()
    get_example_usage()
