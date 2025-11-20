#!/usr/bin/env python3
"""
Create a realistic sample TON_IoT dataset for testing
Based on actual TON_IoT structure
"""

import pandas as pd
import numpy as np

# Create sample TON_IoT data with realistic features
np.random.seed(42)

print("Creating sample TON_IoT dataset...")

n_samples = 5000

# TON_IoT network flow features
data = {
    # Network identifiers
    'src_ip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
    'dst_ip': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
    'sport': np.random.randint(1024, 65535, n_samples),
    'dsport': np.random.choice([22, 80, 443, 3389, 8080, 445], n_samples),
    
    # Protocol (numeric: 6=TCP, 17=UDP, 1=ICMP)
    'proto': np.random.choice([6, 17, 1], n_samples, p=[0.7, 0.25, 0.05]),
    
    # Flow statistics
    'duration': np.random.exponential(5.0, n_samples),
    'bytes': np.random.lognormal(8, 2, n_samples).astype(int),
    'packets': np.random.poisson(50, n_samples),
    'src_bytes': np.random.lognormal(7, 2, n_samples).astype(int),
    'dst_bytes': np.random.lognormal(7, 2, n_samples).astype(int),
    'src_pkts': np.random.poisson(25, n_samples),
    'dst_pkts': np.random.poisson(25, n_samples),
}

df = pd.DataFrame(data)

# Add attack labels (20% attacks)
attack_ratio = 0.2
n_attacks = int(n_samples * attack_ratio)
df['label'] = 0
attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
df.loc[attack_indices, 'label'] = 1

# Add attack types for malicious flows
attack_types = ['ddos', 'dos', 'password', 'scanning', 'xss', 'injection', 'backdoor', 'ransomware', 'mitm']
df['type'] = 'normal'
for idx in attack_indices:
    df.loc[idx, 'type'] = np.random.choice(attack_types)

# Make attacks more distinguishable
for idx in attack_indices:
    attack_type = df.loc[idx, 'type']
    
    if attack_type == 'ddos':
        df.loc[idx, 'packets'] = np.random.randint(1000, 5000)
        df.loc[idx, 'bytes'] = df.loc[idx, 'packets'] * 64
    elif attack_type == 'dos':
        df.loc[idx, 'packets'] = np.random.randint(500, 2000)
        df.loc[idx, 'duration'] = 0.01
    elif attack_type == 'password':
        df.loc[idx, 'dsport'] = 22  # SSH
        df.loc[idx, 'packets'] = np.random.randint(5, 20)
    elif attack_type == 'scanning':
        df.loc[idx, 'packets'] = 1
        df.loc[idx, 'bytes'] = 64
    elif attack_type in ['xss', 'injection']:
        df.loc[idx, 'dsport'] = 80  # HTTP
        df.loc[idx, 'bytes'] = np.random.randint(500, 2000)
    elif attack_type == 'ransomware':
        df.loc[idx, 'bytes'] = np.random.randint(10000, 100000)
    elif attack_type == 'mitm':
        df.loc[idx, 'duration'] = np.random.uniform(30, 300)

# Save to CSV
csv_path = 'TON_IoT_Network.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Created sample dataset: {csv_path}")
print(f"   Total flows: {len(df)}")
print(f"   Normal: {(df['label']==0).sum()}")
print(f"   Attack: {(df['label']==1).sum()}")
print(f"\nAttack type distribution:")
print(df[df['label']==1]['type'].value_counts())
print(f"\nDataset ready for FedSymGraph!")
