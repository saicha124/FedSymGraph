#!/usr/bin/env python3
"""
Automatically update client.py to use TON_IoT dataset
"""

import os
import sys

def find_dataset_file():
    """Find TON_IoT dataset CSV file."""
    possible_names = [
        'TON_IoT_Network.csv',
        'Network-dataset.csv', 
        'Train_Test_Network.csv',
        'ton_iot.csv'
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # Check for any CSV with relevant keywords
    all_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
    for csv in all_csvs:
        if 'ton' in csv.lower() or 'network' in csv.lower():
            print(f"Found possible dataset: {csv}")
            return csv
    
    return None


def update_client():
    """Update client.py to use TON_IoT loader."""
    
    # Find dataset
    dataset_file = find_dataset_file()
    
    if not dataset_file:
        print("❌ No TON_IoT dataset found!")
        print("\nPlease run: python download_toniot.py")
        return False
    
    print(f"✅ Found dataset: {dataset_file}")
    
    # Read client.py
    with open('client.py', 'r') as f:
        content = f.read()
    
    # Check if already updated
    if 'data_loader_toniot' in content:
        print("✅ client.py already configured for TON_IoT")
        return True
    
    # Backup original
    with open('client.py.backup', 'w') as f:
        f.write(content)
    print("📁 Created backup: client.py.backup")
    
    # Update imports
    old_import = "from data_loader import load_local_graphs"
    new_import = "from data_loader_toniot import load_toniot_for_client"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✅ Updated import statement")
    
    # Update loader call
    old_loader = "self.train_loader, self.test_loader = load_local_graphs(client_id)"
    new_loader = f"""self.train_loader, self.test_loader = load_toniot_for_client(
            client_id=client_id,
            csv_path="{dataset_file}",
            batch_size=8,
            flows_per_graph=100
        )"""
    
    if old_loader in content:
        content = content.replace(old_loader, new_loader)
        print("✅ Updated data loader call")
    
    # Write updated file
    with open('client.py', 'w') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("✅ client.py updated successfully!")
    print("="*60)
    print(f"\nDataset: {dataset_file}")
    print("Flows per graph: 100")
    print("Batch size: 8")
    print("\nYou can now run FedSymGraph with real TON_IoT data!")
    print("\nTest it:")
    print("  bash run_demo.sh")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("TON_IoT Client Setup")
    print("="*60)
    print()
    
    success = update_client()
    
    if not success:
        sys.exit(1)
