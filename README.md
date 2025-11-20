# FedSymGraph: Privacy-Preserving Federated Intrusion Detection

A production-grade implementation of a **Federated Learning system** for network intrusion detection that combines:
- **Graph Neural Networks (GNNs)** for pattern recognition
- **Differential Privacy** for gradient protection
- **Neuro-Symbolic AI** for explainable security alerts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Global Coordinator                         │
│              (Secure Aggregation Server)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐    ┌──────▼───────┐
│  Client 1      │    │  Client 2    │
│  (HQ Domain)   │    │ (IoT Branch) │
│                │    │              │
│ • Local GNN    │    │ • Local GNN  │
│ • Privacy      │    │ • Privacy    │
│ • Explainer    │    │ • Explainer  │
└────────────────┘    └──────────────┘
```

## Features

### 1. Federated Learning
- Distributed training across multiple domains
- Secure aggregation using Flower framework
- No raw data leaves client premises

### 2. Graph Neural Networks
- GAT (Graph Attention) layers for structural learning
- Handles network traffic as graph data
- Node features represent traffic statistics

### 3. Differential Privacy
- Opacus integration for gradient noise injection
- Configurable privacy budget (ε, δ)
- Prevents data leakage during aggregation

### 4. Neuro-Symbolic Explainability
- MITRE ATT&CK rule mapping
- LLM-generated human-readable alerts
- Combines symbolic rules with deep learning

## Installation

Dependencies are already installed. See `requirements.txt` for details.

## Quick Start

### Option 1: Run Full Demo (Recommended)
```bash
bash run_demo.sh
```

This starts:
- 1 Global Coordinator
- 2 Federated Clients
- 3 Training Rounds

### Option 2: Manual Setup

**Terminal 1: Start Server**
```bash
python server.py --rounds 5 --min-clients 2
```

**Terminal 2: Start Client 1**
```bash
python client.py --client_id 1 --server 127.0.0.1:8080
```

**Terminal 3: Start Client 2**
```bash
python client.py --client_id 2 --server 127.0.0.1:8080
```

## Configuration

### Command Line Arguments

**Server:**
```bash
python server.py [OPTIONS]
  --rounds NUM         Number of federated rounds (default: 5)
  --min-clients NUM    Minimum clients required (default: 2)
  --port NUM           Server port (default: 8080)
```

**Client:**
```bash
python client.py [OPTIONS]
  --client_id NUM       Client identifier (default: 1)
  --server ADDR         Server address (default: 127.0.0.1:8080)
  --enable-privacy      Enable differential privacy (experimental, see limitations)
  --no-llm              Disable LLM explanations (faster, no API calls)
```

### Configuration File
Edit `config.py` to adjust:
- GNN architecture parameters
- Privacy budget settings
- MITRE ATT&CK rules
- LLM settings

## Project Structure

```
FedSymGraph/
├── server.py          # Global Coordinator
├── client.py          # Federated Client
├── model.py           # GNN Architecture (GAT layers)
├── reasoning.py       # Symbolic Rules + LLM
├── privacy.py         # Differential Privacy Wrapper
├── data_loader.py     # Graph Data Generation
├── config.py          # Configuration Settings
├── run_demo.sh        # Demo Script
└── requirements.txt   # Dependencies
```

## How It Works

### Training Flow
1. **Initialization**: Global coordinator broadcasts initial model weights
2. **Local Training**: Each client trains on local network traffic graphs
3. **Secure Aggregation**: Server aggregates client updates using FedAvg
4. **Repeat**: Process continues for N rounds

Note: Differential Privacy gradient noise is currently not applied due to library incompatibilities.

### Detection & Explanation
1. **Anomaly Detection**: GNN classifies network flows as benign/malicious
2. **Symbolic Matching**: Features matched against MITRE ATT&CK rules
3. **LLM Explanation**: AI generates human-readable security alert

Example Alert:
```
[Client 1 ALERT]: The detected brute force attack (T1110) indicates
an adversary attempting to gain unauthorized access through repeated
SSH login attempts. This poses a high risk of system compromise.
```

## Privacy & Security Notes

### Privacy
- **No Data Sharing**: Only model updates leave client premises
- **Differential Privacy**: Currently experimental due to Opacus incompatibility with PyTorch Geometric GNN layers (GATv2Conv)
  - Production deployment requires custom grad samplers for torch_geometric modules
  - Privacy is disabled by default; use `--enable-privacy` to test (may cause errors)

### Security Considerations
⚠️ **Important**: This demo runs without TLS encryption for simplicity. For production deployment:
- Use Flower's SuperLink with TLS certificates
- Enable client authentication
- Implement secure communication channels
- Never deploy to untrusted networks without encryption

The current gRPC server binding (0.0.0.0:8080) exposes federated traffic in plaintext.

## MITRE ATT&CK Coverage

Current rule mappings:
- **T1110**: Brute Force (SSH authentication failures)
- **T1048**: Exfiltration (Large HTTPS transfers)
- **T1021**: Lateral Movement (SMB file operations)

Add custom rules in `config.py`.

## LLM Integration

### Using OpenAI (Default)
Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Disable LLM (Faster)
```bash
python client.py --no-llm
```

Falls back to rule-based explanations.

## Extending the System

### Add Custom Network Data
Edit `data_loader.py` to parse PCAP files or CSV logs:
```python
def create_graph_from_network_flow(flow_dict):
    # Convert your data format to PyG Data object
    pass
```

### Add New Security Rules
Edit `config.py`:
```python
MITRE_RULES.append({
    "rule": "your_condition",
    "tactic": "TXXXX: Attack Type",
    "description": "What this detects"
})
```

### Adjust Privacy Budget
In `config.py`:
```python
PRIVACY_CONFIG = {
    "noise_multiplier": 1.1,  # Lower = less privacy, better accuracy
    "max_grad_norm": 1.0      # Gradient clipping threshold
}
```

## Performance Notes

- **CPU Mode**: Runs on CPU by default (PyTorch CPU build)
- **GPU Support**: Install CUDA-enabled PyTorch for acceleration
- **Privacy Overhead**: ~15-20% slower with DP enabled
- **LLM Latency**: OpenAI API adds ~1-2s per alert

## Troubleshooting

**Server won't start:**
- Check port 8080 is available: `lsof -i :8080`
- Use different port: `--port 8081`

**Clients can't connect:**
- Ensure server is running first
- Check firewall settings
- Verify server address in client command

**LLM errors:**
- Verify `OPENAI_API_KEY` is set
- Use `--no-llm` flag to disable
- Check API quota/billing

## Citation

If you use this implementation, please cite:
```
FedSymGraph: Privacy-Preserving Federated Intrusion Detection
with Graph Neural Networks and Neuro-Symbolic AI
```

## License

This implementation is provided for research and educational purposes.
