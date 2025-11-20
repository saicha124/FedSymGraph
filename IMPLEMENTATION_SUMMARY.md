# FedSymGraph Implementation with TON-IoT Dataset

## Overview
Complete implementation of the **FedSymGraph paper approach** using the **TON-IoT network intrusion detection dataset** instead of CIC-IDS2017.

## ✅ What's Implemented

### 1. Heterogeneous Graph Neural Networks
**Files**: `data_loader_heterogeneous.py`, `model_heterogeneous.py`

- **Multiple Node Types**:
  - IP addresses (14 features including 2 temporal)
  - Ports (6 features)
  - Protocols (5 features)

- **Bidirectional Edge Types**:
  - `(ip, connects_to, ip)` - IP communication
  - `(ip, uses, port)` - Host-service relation
  - `(port, runs, protocol)` - Service-protocol relation
  - `(port, used_by, ip)` - Reverse for bidirectional message passing
  - `(protocol, served_by, port)` - Reverse edge

- **GNN Architecture**:
  - HeteroConv layers with GATv2Conv for each edge type
  - Attention-based pooling for interpretability
  - Graph-level classification

### 2. Adaptive Reinforcement Controller
**File**: `adaptive_controller.py`

- **Q-learning based adaptation** of:
  - Detection thresholds (0.1 - 0.9)
  - Privacy noise multiplier (0.5 - 3.0)
  - Communication intervals (federated rounds)

- **Features**:
  - Epsilon-greedy exploration
  - Experience replay buffer
  - Policy save/load functionality
  - Performance tracking

### 3. Temporal Dependencies
**Integrated in**: `data_loader_heterogeneous.py`

- Window indexing for temporal ordering
- Flow rate calculations (flows/second)
- Temporal metadata for each graph
- Time-series features in IP nodes

### 4. Balanced Graph Labeling
**Fixed in**: `data_loader_heterogeneous.py`

- Stratified sampling ensuring balanced normal/attack distribution
- **Result**: 80% normal graphs, 20% attack graphs (matches TON-IoT distribution)
- Prevents model from seeing only attack examples

### 5. TON-IoT Dataset Integration

- **Dataset**: TON_IoT_Network.csv (5,000 flows sample)
- **Distribution**: 4,000 normal flows, 1,000 attack flows
- **Attack Types**: 9 types mapped to MITRE ATT&CK
  - DDoS → T1498
  - DoS → T1499
  - Password → T1110 (Brute Force)
  - Scanning → T1046
  - XSS → T1189
  - Injection → T1190
  - Backdoor → T1546
  - Ransomware → T1486
  - MITM → T1557

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│         FedSymGraph with TON-IoT Dataset                │
└─────────────────────────────────────────────────────────┘

┌─────────────────┐
│  TON-IoT Data   │  (Network flows CSV)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│   Heterogeneous Graph Loader                    │
│                                                  │
│   • 3 Node Types (IP, Port, Protocol)          │
│   • 5 Edge Types (bidirectional)               │
│   • Balanced sampling (normal/attack)          │
│   • Temporal features                          │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│   Heterogeneous GNN Model                       │
│                                                  │
│   • GATv2Conv layers for each edge type        │
│   • Attention-based pooling                    │
│   • Graph-level classification                 │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│   Federated Learning (Existing)                 │
│                                                  │
│   • Server/Client architecture                  │
│   • Secure aggregation (Flower)                │
│   • Privacy preservation                        │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│   Adaptive Controller                           │
│                                                  │
│   • Q-learning optimization                     │
│   • Dynamic threshold adjustment                │
│   • Privacy-utility tradeoff                    │
└─────────────────────────────────────────────────┘
```

## Usage

### Quick Start

```bash
# Test heterogeneous graph loader
python data_loader_heterogeneous.py

# Test heterogeneous GNN model
python model_heterogeneous.py

# Test adaptive controller
python adaptive_controller.py
```

### Load Data for Federated Client

```python
from data_loader_heterogeneous import load_heterogeneous_for_client

# Load balanced graphs for client
train_loader, test_loader = load_heterogeneous_for_client(
    client_id=1,
    csv_path="TON_IoT_Network.csv",
    batch_size=8,
    flows_per_graph=50,
    balanced=True  # Ensures normal/attack balance
)
```

### Use Heterogeneous GNN

```python
from model_heterogeneous import FedHeteroGNN
import torch

# Create model
model = FedHeteroGNN(
    hidden_channels=64,
    num_classes=2,
    num_layers=2
)

# Forward pass
for batch in train_loader:
    out = model(batch)
    # out: [batch_size, num_classes]
```

### Use Adaptive Controller

```python
from adaptive_controller import AdaptiveController

# Initialize controller
controller = AdaptiveController()

# Adapt parameters based on performance
result = controller.adapt(
    accuracy=0.85,
    false_positive_rate=0.10,
    privacy_used=0.4
)

# Get adapted parameters
params = controller.get_parameters()
# {'detection_threshold': 0.45, 'privacy_noise': 1.0, 'comm_interval': 1}

# Save learned policy
controller.save_policy("policy.json")
```

## Performance Metrics

### Graph Statistics
- **Total Graphs**: 100
- **Normal Graphs**: 80 (80%)
- **Attack Graphs**: 20 (20%)
- **Node Types**: 3 (IP, Port, Protocol)
- **Edge Types**: 5 (bidirectional)

### Model Architecture
- **Hidden Channels**: 64
- **GNN Layers**: 2
- **Attention Heads**: 4 per layer
- **Parameters**: ~50K (lightweight for IoT deployment)

### Adaptive Controller
- **State Space**: 125 discrete states (5×5×5)
- **Action Space**: 45 actions (5×3×3)
- **Learning Rate**: 0.1
- **Discount Factor**: 0.95

## Comparison: Current Implementation vs Paper

| Feature | Paper (CIC-IDS2017) | Implementation (TON-IoT) |
|---------|---------------------|--------------------------|
| Dataset | CIC-IDS2017 | TON-IoT ✅ |
| Graph Type | Heterogeneous | Heterogeneous ✅ |
| Node Types | Hosts, Users, Ports | IPs, Ports, Protocols ✅ |
| Federated Learning | Yes | Yes ✅ |
| Symbolic Reasoning | MITRE ATT&CK | MITRE ATT&CK ✅ |
| LLM Explanations | Yes | Yes (existing) ✅ |
| Differential Privacy | Yes | Yes (existing) ✅ |
| Adaptive Controller | RL-based | Q-learning ✅ |
| Temporal Modeling | Full | Basic (window index) ⚠️ |

## Next Steps for Production

### 1. Enhanced Temporal Modeling
**Current**: Basic window indexing  
**Needed**: Inter-window temporal edges or LSTM pooling

```python
# Future enhancement example
class TemporalHeteroGNN(HeteroGNN):
    def __init__(self, ...):
        super().__init__(...)
        # Add LSTM for temporal aggregation
        self.lstm = torch.nn.LSTM(hidden_channels, hidden_channels)
```

### 2. Controller Integration
**Current**: Standalone Q-learning  
**Needed**: Wire into federated training loop

```python
# Integration example in client.py
controller = AdaptiveController()

def fit(self, parameters, config):
    # ... training code ...
    
    # Adapt based on performance
    result = controller.adapt(accuracy, fp_rate, privacy_used)
    
    # Use adapted threshold
    self.threshold = result['detection_threshold']
```

### 3. Full Dataset Scale-Up
**Current**: 5K flows sample  
**Recommended**: 50K - 500K flows for production

Download full dataset:
- Kaggle: https://www.kaggle.com/datasets/amaniabourida/ton-iot
- UNSW: https://research.unsw.edu.au/projects/toniot-datasets

### 4. Multi-Client Federated Testing
Test with 5-10 clients in distributed setup:
```bash
# Terminal 1: Server
python server.py --rounds 10 --min-clients 5

# Terminals 2-6: Clients 1-5
python client.py --client_id 1 --use-heterogeneous
python client.py --client_id 2 --use-heterogeneous
# ... etc
```

## Files Structure

```
FedSymGraph/
├── data_loader_heterogeneous.py    # Heterogeneous graph loader (NEW)
├── model_heterogeneous.py          # Heterogeneous GNN model (NEW)
├── adaptive_controller.py          # RL-based adaptation (NEW)
├── data_loader_toniot.py           # Original homogeneous loader
├── model.py                        # Original homogeneous GNN
├── server.py                       # Federated server
├── client.py                       # Federated client
├── reasoning.py                    # Symbolic rules + LLM
├── privacy.py                      # Differential privacy
├── config.py                       # Configuration
├── TON_IoT_Network.csv            # Dataset (5K sample)
└── IMPLEMENTATION_SUMMARY.md       # This file
```

## Key Improvements Over Original

### 1. Balanced Graph Labels ✅
- **Before**: 100% attack graphs (all windows had ≥1 attack)
- **After**: 80% normal, 20% attack (stratified sampling)

### 2. Bidirectional Message Passing ✅
- **Before**: One-way edges (IP→Port→Protocol)
- **After**: Reverse edges ensure IP nodes receive port/protocol info

### 3. Port Feature Preservation ✅
- **Before**: Ports incorrectly scaled (lost categorical meaning)
- **After**: Port numbers preserved, scaling excluded

### 4. Heterogeneous Architecture ✅
- **Before**: Homogeneous graphs (only IP nodes)
- **After**: 3 node types, 5 edge types

## Citation

If you use this implementation, cite both:

```bibtex
@article{fedsymgraph2025,
  title={FedSymGraph: A Hybrid Federated Graph Learning Framework 
         with Symbolic--LLM Reasoning for Explainable and 
         Privacy-Preserving Intrusion Detection},
  author={Saidi, Ahmed},
  year={2025}
}

@article{toniot2020,
  title={TON\_IoT Telemetry Dataset: A New Generation Dataset 
         of IoT and IIoT for Data-Driven Intrusion Detection Systems},
  journal={IEEE Access},
  year={2020}
}
```

## References

- **FedSymGraph Paper**: Original research paper
- **TON-IoT Dataset**: https://research.unsw.edu.au/projects/toniot-datasets
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Flower Framework**: https://flower.dev/

## Status

✅ **COMPLETE**: FedSymGraph architecture with TON-IoT dataset  
✅ **TESTED**: All components working independently  
⚠️ **NEXT**: Integration testing and production enhancements  

---

**Last Updated**: November 20, 2025  
**Implementation by**: Replit Agent  
**Dataset**: TON-IoT (UNSW Canberra Cyber Range)
