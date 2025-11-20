#!/usr/bin/env python3
"""
TON_IoT Dataset Downloader
Downloads the TON_IoT network dataset for FedSymGraph
"""

import os
import sys

def download_from_kaggle():
    """Download TON_IoT dataset from Kaggle using kaggle API."""
    try:
        import kaggle
        print("Kaggle API found. Attempting download...")
        print("\nDownloading TON_IoT dataset from Kaggle...")
        print("This may take a few minutes (dataset is ~500MB)...\n")
        
        kaggle.api.dataset_download_files(
            'amaniabourida/ton-iot',
            path='.',
            unzip=True
        )
        
        print("\n✅ Dataset downloaded successfully!")
        
        # Find the CSV file
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'network' in f.lower()]
        if csv_files:
            print(f"Found dataset file: {csv_files[0]}")
            return csv_files[0]
        else:
            print("Dataset downloaded but CSV not found in expected location.")
            return None
            
    except ImportError:
        print("❌ Kaggle API not installed.")
        print("\nTo install:")
        print("  pip install kaggle")
        print("\nThen configure your API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. This downloads kaggle.json")
        print("  4. Upload kaggle.json to this project")
        return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Manual download")
        print("  1. Go to: https://www.kaggle.com/datasets/amaniabourida/ton-iot")
        print("  2. Click 'Download' button")
        print("  3. Upload the CSV file to this project")
        return None


def download_from_huggingface():
    """Download TON_IoT dataset from Hugging Face."""
    try:
        from datasets import load_dataset
        print("Downloading TON_IoT from Hugging Face...")
        print("This may take a few minutes...\n")
        
        dataset = load_dataset("codymlewis/TON_IoT_network", split="train")
        
        # Convert to pandas and save
        df = dataset.to_pandas()
        csv_path = "TON_IoT_Network.csv"
        
        # Sample if too large
        if len(df) > 100000:
            print(f"Dataset has {len(df)} rows. Sampling 100,000 for demo...")
            df = df.sample(100000, random_state=42)
        
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Dataset saved to: {csv_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        return csv_path
        
    except ImportError:
        print("❌ Hugging Face datasets library not installed.")
        print("\nTo install:")
        print("  pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def check_existing_dataset():
    """Check if dataset already exists in the project."""
    possible_names = [
        'TON_IoT_Network.csv',
        'Network-dataset.csv',
        'Train_Test_Network.csv',
        'ton_iot.csv'
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            print(f"✅ Found existing dataset: {name}")
            return name
    
    # Check for any CSV with 'ton' or 'network' in name
    all_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
    for csv in all_csvs:
        if 'ton' in csv.lower() or 'network' in csv.lower():
            print(f"✅ Found possible dataset: {csv}")
            response = input(f"Is this the TON_IoT dataset? (y/n): ").strip().lower()
            if response == 'y':
                return csv
    
    return None


def main():
    print("="*60)
    print("TON_IoT Dataset Downloader for FedSymGraph")
    print("="*60)
    print()
    
    # Check if dataset already exists
    existing = check_existing_dataset()
    if existing:
        print(f"\nDataset ready: {existing}")
        print("\nYou can now run FedSymGraph with TON_IoT data!")
        return existing
    
    print("Dataset not found. Choose download method:\n")
    print("1. Hugging Face (recommended - easiest)")
    print("2. Kaggle (requires API key)")
    print("3. Manual download instructions")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        return download_from_huggingface()
    elif choice == '2':
        return download_from_kaggle()
    elif choice == '3':
        print("\n" + "="*60)
        print("Manual Download Instructions")
        print("="*60)
        print("\nOption A - Kaggle (Full Dataset):")
        print("  1. Go to: https://www.kaggle.com/datasets/amaniabourida/ton-iot")
        print("  2. Sign in to Kaggle")
        print("  3. Click 'Download' button")
        print("  4. Extract the ZIP file")
        print("  5. Upload the CSV file to this Replit project")
        print("\nOption B - Hugging Face (Sampled):")
        print("  1. Install: pip install datasets")
        print("  2. Run this script again and choose option 1")
        print("\n" + "="*60)
        return None
    else:
        print("Exiting.")
        return None


if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "="*60)
        print("✅ SUCCESS! Dataset is ready")
        print("="*60)
        print(f"\nDataset file: {result}")
        print("\nNext step: Update client.py to use TON_IoT data")
        print("Run: python setup_toniot_client.py")
    else:
        print("\n⚠️  Dataset not downloaded")
        print("Please download manually or try again")
        sys.exit(1)
