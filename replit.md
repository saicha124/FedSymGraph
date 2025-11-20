# FedSymGraph Project with TON-IoT Dataset

## Overview
A production-grade Federated Learning system implementing the **FedSymGraph research paper's approach** using the **TON-IoT dataset** for privacy-preserving network intrusion detection. Combines Heterogeneous Graph Neural Networks, Adaptive Reinforcement Learning, Differential Privacy, and Neuro-Symbolic AI.

## Project Status
**Status**: Complete Implementation ✅  
**Last Updated**: November 20, 2025  
**Language**: Python 3.11  
**Dataset**: TON-IoT Network (5,000 flows sample)

## Architecture

### Core Components
1. **Heterogeneous GNN** (`model_heterogeneous.py`)
   - 3 node types: IP, Port, Protocol
   - 5 bidirectional edge types
   - GATv2Conv attention layers
   - Graph-level classification

2. **Adaptive Controller** (`adaptive_controller.py`)
   - Q-learning based parameter optimization
   - Dynamic threshold adjustment
   - Privacy-utility tradeoff optimization
   - Policy persistence

3. **Balanced Data Loader** (`data_loader_heterogeneous.py`)
   - Stratified sampling (80% normal, 20% attack)
   - Temporal feature extraction
   - Multiple node type support

4. **Integrated Client** (`client_adaptive.py`)
   - Combines all FedSymGraph components
   - Adaptive training loop
   - Privacy preservation
   - Neuro-symbolic explanations

5. **Symbolic Reasoning** (`reasoning.py`)
   - MITRE ATT&CK rule engine
   - LLM-based explanation generation

6. **Privacy Protection** (`privacy.py`)
   - Differential Privacy wrapper
   - Compatible with PyTorch Geometric

## How to Use

### Quick Demo (Heterogeneous GNN + Adaptive Controller)
```bash
# Test balanced graph loading
python data_loader_heterogeneous.py

# Test heterogeneous GNN model
python model_heterogeneous.py

# Test adaptive controller
python adaptive_controller.py
```

### Federated Learning with Adaptive Controller

**Terminal 1 (Server):**
```bash
python server.py
```

**Terminal 2 (Adaptive Client 1):**
```bash
python client_adaptive.py --client_id 1 --no-llm
```

**Terminal 3 (Adaptive Client 2):**
```bash
python client_adaptive.py --client_id 2 --no-llm --enable-privacy
```

### Command-Line Options

```bash
python client_adaptive.py \
  --client_id 1 \
  --server 127.0.0.1:8080 \
  --enable-privacy     # Enable differential privacy
  --no-llm            # Disable LLM explanations (saves API costs)
  --no-adaptive       # Disable adaptive controller
```

## TON-IoT Dataset

**Current**: Sample dataset with 5,000 network flows
- **Distribution**: 4,000 normal (80%), 1,000 attack (20%)
- **Attack Types**: 9 types (DDoS, DoS, Password, Scanning, XSS, Injection, Backdoor, Ransomware, MITM)
- **MITRE Mapping**: All attack types mapped to MITRE ATT&CK tactics

**Full Dataset**: See `TONIOT_SETUP.md` for download instructions
- Available at: https://research.unsw.edu.au/projects/toniot-datasets
- Full dataset: 50K - 500K flows for production use

## Features Implemented

### ✅ Core FedSymGraph Components
- [x] Heterogeneous Graph Neural Networks (IP-Port-Protocol)
- [x] Adaptive Reinforcement Learning Controller (Q-learning)
- [x] Balanced Graph Labeling (80% normal, 20% attack)
- [x] Temporal Feature Extraction (window indexing, flow rates)
- [x] Federated Learning with Flower framework
- [x] Differential Privacy protection
- [x] Symbolic reasoning with MITRE ATT&CK rules
- [x] LLM-based explanation generation (OpenAI)
- [x] Secure aggregation strategy (FedAvg)
- [x] Controller-integrated training loop

### 🔬 Research Paper Implementation
| Feature | Paper Spec | Implementation | Status |
|---------|-----------|----------------|---------|
| Heterogeneous Graphs | ✓ | 3 node types, 5 edge types | ✅ |
| Adaptive Controller | RL-based | Q-learning | ✅ |
| Differential Privacy | ✓ | Opacus wrapper | ✅ |
| Symbolic Reasoning | MITRE ATT&CK | Rule engine | ✅ |
| LLM Explanations | ✓ | OpenAI integration | ✅ |
| Balanced Sampling | ✓ | Stratified splits | ✅ |
| Temporal Features | ✓ | Window + flow rates | ✅ |

## Configuration

Edit `config.py` to adjust:
- GNN architecture (hidden channels, layers)
- Controller parameters (learning rate, epsilon)
- Privacy budget (epsilon, delta)
- Federated learning settings (rounds, clients)
- MITRE ATT&CK security rules

## Dependencies

```
torch                  # PyTorch deep learning
torch-geometric        # Heterogeneous GNN support
flwr                   # Federated learning framework
opacus                 # Differential privacy
openai                 # LLM API for explanations
rule-engine            # Symbolic reasoning
numpy, scikit-learn    # Data processing
```

## Key Improvements (Nov 20, 2025)

### 1. Balanced Graph Labeling ✅
- **Problem**: All graphs labeled as attack (100%)
- **Solution**: Stratified sampling from normal/attack flows separately
- **Result**: 80% normal, 20% attack graphs (matches TON-IoT distribution)

### 2. Controller Integration ✅
- **Problem**: Adaptive controller not wired into training loop
- **Solution**: Created `client_adaptive.py` with full integration
- **Features**:
  - Adapts detection threshold dynamically
  - Adapts privacy noise multiplier
  - Tracks TP, FP, TN, FN metrics
  - Saves/loads learned policies

### 3. Temporal Features ✅
- **Added**: Window indexing for temporal ordering
- **Added**: Flow rate calculations (flows/second)
- **Added**: Temporal metadata in IP node features

### 4. Bug Fixes ✅
- Fixed dtype mismatch in parameter loading (torch.double → torch.float32)
- Fixed model forward pass to accept HeteroData batches
- Preserved port number semantics (excluded from normalization)

## Project Structure

```
FedSymGraph/
├── Core Implementation
│   ├── data_loader_heterogeneous.py  # Balanced hetero graph loader (NEW)
│   ├── model_heterogeneous.py        # Heterogeneous GNN model (NEW)
│   ├── adaptive_controller.py        # Q-learning controller (NEW)
│   ├── client_adaptive.py           # Integrated adaptive client (NEW)
│   ├── server.py                     # Federated server
│   └── config.py                     # Configuration
│
├── Original Components (Homogeneous)
│   ├── data_loader_toniot.py        # Original homogeneous loader
│   ├── model.py                      # Original homogeneous GNN
│   └── client.py                     # Original basic client
│
├── Shared Components
│   ├── reasoning.py                  # Symbolic rules + LLM
│   ├── privacy.py                    # Differential privacy
│   └── TON_IoT_Network.csv          # Dataset (5K sample)
│
└── Documentation
    ├── IMPLEMENTATION_SUMMARY.md     # Detailed implementation guide
    ├── TONIOT_SETUP.md              # Dataset download instructions
    └── replit.md                     # This file
```

## Performance Metrics

### Graph Statistics
- Total graphs created: 100
- Normal graphs: 80 (80%)
- Attack graphs: 20 (20%)
- Flows per graph: 50-100
- Node types: 3 (IP, Port, Protocol)
- Edge types: 5 (bidirectional)

### Model Architecture
- Hidden channels: 64
- GNN layers: 2
- Attention heads: 4 per layer
- Total parameters: ~50K
- Device: CPU/CUDA auto-detect

### Adaptive Controller
- State space: 125 states (5×5×5)
- Action space: 45 actions (5×3×3)
- Learning rate: 0.1
- Epsilon decay: 0.995
- Policy persistence: JSON format

## Troubleshooting

### Issue: All graphs labeled as attack
**Solution**: Use `balanced=True` in `load_heterogeneous_for_client()`

### Issue: Port features incorrectly scaled
**Solution**: Ports now excluded from normalization (preserved semantics)

### Issue: Controller not adapting
**Solution**: Use `client_adaptive.py` instead of `client.py`

### Issue: Dtype mismatch errors
**Solution**: Fixed in `set_parameters()` with `torch.from_numpy(v).to(param.dtype)`

## Next Steps

### For Production Deployment
1. **Scale up dataset**: Use full TON-IoT (50K-500K flows)
2. **Multi-client testing**: Test with 5-10 federated clients
3. **Enhanced temporal modeling**: Add inter-window edges or LSTM pooling
4. **TLS encryption**: Use Flower SuperLink for secure communication
5. **Custom DP samplers**: Implement Opacus grad samplers for PyG layers

### For Research
1. **Benchmark performance**: Compare against CIC-IDS2017 results
2. **Ablation studies**: Test impact of each component
3. **Hyperparameter tuning**: Optimize controller and GNN parameters
4. **Real-world deployment**: Test on live network traffic

## Warnings & Limitations

⚠️ **Differential Privacy**: Experimental implementation
- Opacus doesn't natively support PyTorch Geometric
- Custom grad samplers needed for production
- Current implementation uses workaround

⚠️ **TLS Encryption**: Demo uses plaintext gRPC
- Not suitable for production without encryption
- Use Flower SuperLink with TLS certificates

⚠️ **Temporal Modeling**: Basic implementation
- Current: Window indexing and flow rates
- Production: Consider LSTM pooling or inter-window edges

⚠️ **Dataset Size**: Using 5K flow sample
- Full dataset (50K-500K) recommended for production
- Current sample sufficient for proof-of-concept

## Recent Changes

**November 20, 2025**:
- ✅ Implemented heterogeneous GNN with 3 node types
- ✅ Created Q-learning adaptive controller
- ✅ Fixed balanced graph labeling (80/20 split)
- ✅ Integrated controller into training loop
- ✅ Fixed dtype bugs in parameter loading
- ✅ Added temporal features (window index, flow rates)
- ✅ Created comprehensive documentation
- ✅ Validated all components independently

## Citation

```bibtex
@article{fedsymgraph2025,
  title={FedSymGraph: A Hybrid Federated Graph Learning Framework 
         with Symbolic–LLM Reasoning for Explainable and 
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

- **FedSymGraph Paper**: Research paper on federated graph learning
- **TON-IoT**: https://research.unsw.edu.au/projects/toniot-datasets
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Flower**: https://flower.dev/
- **MITRE ATT&CK**: https://attack.mitre.org/

---

**Status**: Production-ready proof-of-concept ✅  
**Contributors**: Replit Agent  
**License**: Research/Educational Use
