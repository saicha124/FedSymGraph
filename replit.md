# FedSymGraph Project

## Overview
A production-grade Federated Learning system for privacy-preserving network intrusion detection. Combines Graph Neural Networks (GNNs), Differential Privacy, and Neuro-Symbolic AI for explainable security alerts.

## Project Status
**Status**: Complete MVP Implementation
**Last Updated**: November 20, 2025
**Language**: Python 3.11

## Architecture
- **Framework**: Flower (Federated Learning)
- **ML Model**: PyTorch Geometric (Graph Neural Networks with GAT layers)
- **Privacy**: Opacus (Differential Privacy)
- **Explainability**: Rule Engine + OpenAI (Neuro-Symbolic AI)

## Key Components
1. `server.py` - Global coordinator for federated aggregation
2. `client.py` - Federated client with local training and privacy
3. `model.py` - GNN architecture (GAT layers)
4. `reasoning.py` - MITRE ATT&CK rule engine + LLM explanations
5. `privacy.py` - Differential privacy wrapper
6. `data_loader.py` - Synthetic network graph data generator

## How to Use

### Quick Demo
```bash
bash run_demo.sh
```

### Manual Mode
**Terminal 1 (Server):**
```bash
python server.py
```

**Terminal 2 (Client 1):**
```bash
python client.py --client_id 1 --no-llm
```

**Terminal 3 (Client 2):**
```bash
python client.py --client_id 2 --no-llm
```

## TON_IoT Dataset

**Currently Using**: Sample TON_IoT dataset (5,000 flows, 9 attack types)
- 80% normal traffic, 20% attacks
- Attack types: DDoS, DoS, Password cracking, Scanning, XSS, Injection, Backdoor, Ransomware, MITM

**To use full dataset**: See `TONIOT_SETUP.md` for download instructions

## Features Implemented
- ✅ Federated Learning with Flower framework
- ✅ Graph Neural Networks (GAT architecture)
- ⚠️ Differential Privacy protection (experimental - disabled by default due to Opacus/PyG incompatibility)
- ✅ Symbolic reasoning with MITRE ATT&CK rules
- ✅ LLM-based explanation generation (OpenAI integration)
- ✅ Secure aggregation strategy (FedAvg)
- ✅ Synthetic network traffic data generation
- ✅ Integration test suite

## Configuration
Edit `config.py` to adjust:
- GNN architecture parameters
- Privacy budget (epsilon, delta)
- Federated learning settings
- MITRE ATT&CK security rules

## Dependencies
- torch (PyTorch)
- torch-geometric (Graph Neural Networks)
- flwr (Federated Learning)
- opacus (Differential Privacy)
- openai (LLM API)
- rule-engine (Symbolic reasoning)
- numpy, scikit-learn

## Recent Changes
- Nov 20, 2025: Initial implementation complete
  - Created all core modules
  - Set up federated architecture
  - Disabled DP by default (Opacus incompatible with PyTorch Geometric)
  - Added neuro-symbolic explainability
  - Generated synthetic data for testing
  - Added security warnings for TLS encryption
  - Created integration test suite
  - Updated documentation with limitations

## Notes
- Server runs on port 8080 by default
- LLM explanations require OPENAI_API_KEY (use `--no-llm` flag to disable)
- ⚠️ Differential Privacy is experimental and disabled by default
  - Opacus doesn't support torch_geometric GNN layers
  - Production requires custom grad samplers
- ⚠️ Current demo uses plaintext gRPC (no TLS)
  - Not suitable for production without encryption
  - Use Flower SuperLink with TLS for deployment
- System simulates distributed enterprise scenario (HQ + IoT Branch)
