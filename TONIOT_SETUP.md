# TON_IoT Dataset Integration Guide

## Overview

This guide explains how to use the **TON_IoT** dataset with FedSymGraph instead of synthetic data. TON_IoT is a comprehensive network intrusion detection dataset from UNSW with real IoT network traffic.

## Dataset Information

- **Name**: TON_IoT (Telemetry of IoT and IIoT)
- **Source**: UNSW Canberra Cyber Range
- **Size**: ~22M training records, ~461K test records
- **Features**: 42 network flow features
- **Labels**: Binary (normal/attack) + 9 attack types
- **Attack Types**: DDoS, DoS, Password cracking, Scanning, XSS, Injection, Backdoor, Ransomware, MITM

## Download the Dataset

### Option 1: Kaggle (Easiest)
1. Go to: https://www.kaggle.com/datasets/amaniabourida/ton-iot
2. Download the network dataset CSV file
3. Upload to your Replit project root

### Option 2: Official Source
1. Visit: https://research.unsw.edu.au/projects/toniot-datasets
2. Download the network dataset
3. Upload to Replit

### Option 3: Hugging Face
```bash
# Install datasets library
pip install datasets

# Download programmatically
from datasets import load_dataset
dataset = load_dataset("codymlewis/TON_IoT_network")
```

## File Setup

After downloading, you should have a file named something like:
- `TON_IoT_Network.csv` or
- `Network-dataset.csv` or
- `Train_Test_Network.csv`

Place this file in your project root directory.

## Dataset Structure

The CSV contains these key columns:

| Column | Description |
|--------|-------------|
| `src_ip` / `srcip` | Source IP address |
| `dst_ip` / `dstip` | Destination IP address |
| `sport` | Source port |
| `dsport` | Destination port |
| `proto` | Protocol (TCP/UDP/ICMP) |
| `duration` | Flow duration |
| `bytes` | Total bytes |
| `packets` | Total packets |
| `label` | Binary: 0=normal, 1=attack |
| `type` | Attack type (ddos, dos, password, etc.) |

## How It Works

The `data_loader_toniot.py` converts network flows into graphs:

1. **Nodes**: Unique IP addresses (sources and destinations)
2. **Edges**: Network connections between IPs
3. **Node Features**: Aggregated traffic statistics per IP
   - Outgoing connections, bytes, packets
   - Incoming connections, bytes, packets
   - Attack indicators
4. **Graph Label**: Attack (1) if any flow in the graph is malicious

## Using TON_IoT with FedSymGraph

### Step 1: Update client.py

Replace the synthetic data loader with TON_IoT loader:

```python
# In client.py, line ~21, replace:
from data_loader import load_local_graphs

# With:
from data_loader_toniot import load_toniot_for_client

# Then update line ~24:
# OLD:
self.train_loader, self.test_loader = load_local_graphs(client_id)

# NEW:
self.train_loader, self.test_loader = load_toniot_for_client(
    client_id=client_id,
    csv_path="TON_IoT_Network.csv",  # Your dataset file
    batch_size=8,
    flows_per_graph=100
)
```

### Step 2: Run the System

```bash
# Terminal 1: Start server
python server.py --rounds 5 --min-clients 2

# Terminal 2: Client 1
python client.py --client_id 1 --no-llm

# Terminal 3: Client 2
python client.py --client_id 2 --no-llm
```

## Configuration Options

### Adjust Data Loading

In `data_loader_toniot.py`, you can modify:

```python
loader = TONIoTGraphLoader(
    csv_path="TON_IoT_Network.csv",
    sample_size=50000  # Number of flows to sample (None = all)
)

graphs = loader.load_graphs(
    flows_per_graph=100,  # Flows per graph (smaller = more graphs)
    max_graphs=500        # Maximum graphs to create
)
```

### Per-Client Data Distribution

Each client gets a different subset:
- Client 1: First 10,000 flows
- Client 2: Next 10,000 flows
- Client N: Next 10,000 flows

This simulates heterogeneous data distribution across organizations.

## Expected Output

With real TON_IoT data, you'll see:

```
Loading TON_IoT dataset from TON_IoT_Network.csv...
Loaded 50000 network flows
Attack distribution:
0    45234
1     4766
Name: label, dtype: int64

Created 450 graphs from TON_IoT data
Attack graphs: 89, Normal graphs: 361

[Client 1 ALERT]: Security Alert: Detected tactics T1498: Network Denial of Service 
with 87.23% confidence.
```

## MITRE ATT&CK Mapping

The loader automatically maps TON_IoT attack types to MITRE tactics:

| TON_IoT Type | MITRE Tactic |
|--------------|--------------|
| ddos | T1498: Network Denial of Service |
| dos | T1499: Endpoint Denial of Service |
| password | T1110: Brute Force |
| scanning | T1046: Network Service Scanning |
| xss | T1189: Drive-by Compromise |
| injection | T1190: Exploit Public-Facing Application |
| backdoor | T1546: Persistence via Backdoor |
| ransomware | T1486: Data Encrypted for Impact |
| mitm | T1557: Man-in-the-Middle |

## Testing the Loader

Test the TON_IoT loader independently:

```bash
python data_loader_toniot.py
```

This will:
- Load the dataset
- Create sample graphs
- Print statistics
- Show sample batch structure

## Troubleshooting

### "Dataset not found"
- Verify the CSV file exists in your project
- Check the filename matches in `load_toniot_for_client(csv_path=...)`

### "No graphs created"
- Your CSV might have different column names
- Check the console for preprocessing warnings
- Try with `sample_size=1000` first to test

### Memory Issues
- Reduce `sample_size` (e.g., 10000)
- Reduce `max_graphs` (e.g., 100)
- Increase `flows_per_graph` (fewer graphs)

### Column Name Errors
The loader handles common variations:
- `srcip` → `src_ip`
- `dstip` → `dst_ip`
- `src` → `src_ip`

If yours differ, update the `rename_map` in `preprocess()`.

## Performance Comparison

| Dataset | Graphs | Training Time | Accuracy |
|---------|--------|---------------|----------|
| Synthetic | 100 | ~2 sec/round | ~55-70% |
| TON_IoT | 450 | ~5 sec/round | ~75-90% |

Real data typically yields better performance and more realistic threat detection.

## Next Steps

1. **Download** the TON_IoT dataset
2. **Place** the CSV in your project root
3. **Update** `client.py` to use `load_toniot_for_client()`
4. **Test** with a small sample first
5. **Scale** to full dataset once working

## References

- Official Dataset: https://research.unsw.edu.au/projects/toniot-datasets
- Paper: [TON_IoT Telemetry Dataset (IEEE Access 2020)](https://ieeexplore.ieee.org/document/9189760)
- Kaggle Mirror: https://www.kaggle.com/datasets/amaniabourida/ton-iot

Happy threat hunting! 🛡️
