# FedSymGraph Setup & Testing Guide

## Quick Start

The server is already running in the background. To test the system:

### Option 1: Manual Client Testing

**Terminal 1: Start Client 1**
```bash
python client.py --client_id 1 --server 127.0.0.1:8080 --no-llm
```

**Terminal 2: Start Client 2**  
```bash
python client.py --client_id 2 --server 127.0.0.1:8080 --no-llm
```

The server will coordinate 3 federated learning rounds between the clients.

### Option 2: Run Demo Script
```bash
bash run_demo.sh
```

This automatically starts the server and 2 clients.

### Option 3: Run Integration Test
```bash
python test_integration.py
```

This runs an automated test with 2 rounds and 2 clients.

## What You'll See

### Server Output
```
FedSymGraph Global Coordinator
Port: 8080
Federated Rounds: 3
Minimum Clients Required: 2
Waiting for federated clients to connect...

INFO: [INIT] Requesting initial parameters from one random client
INFO: [ROUND 1] fit_round: strategy sampled 2 clients
INFO: [ROUND 1] aggregate_fit: received 2 results
INFO: [ROUND 1] evaluate_round: strategy sampled 2 clients
...
```

### Client Output
```
============================================================
FedSymGraph Client 1 Starting
Server: 127.0.0.1:8080
Privacy: Disabled
LLM Explanations: Disabled
============================================================

[Client 1] Training loss: 0.6932
[Client 1] Evaluation - Loss: 0.6931, Accuracy: 0.5000

[Client 1 ALERT]: Security Alert: Detected tactics T1110: Brute Force 
with 99.00% confidence.
```

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│          Global Coordinator (Port 8080)      │
│                                              │
│  - Aggregates model updates (FedAvg)        │
│  - Coordinates 3 training rounds            │
│  - No raw data access                       │
└──────────────┬──────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼──────┐  ┌──▼───────────┐
│   Client 1   │  │   Client 2   │
│              │  │              │
│ • GNN Model  │  │ • GNN Model  │
│ • Local Data │  │ • Local Data │
│ • Explainer  │  │ • Explainer  │
└──────────────┘  └──────────────┘
```

## Key Components

1. **Server** (`server.py`): Federated coordinator using Flower's FedAvg strategy
2. **Client** (`client.py`): Local training with GNN model and explainability
3. **Model** (`model.py`): Graph Attention Network (GAT) for intrusion detection
4. **Reasoning** (`reasoning.py`): MITRE ATT&CK rules + LLM explanations
5. **Data Loader** (`data_loader.py`): Synthetic network graph generation

## Configuration

Edit `config.py` to modify:
- Number of federated rounds
- GNN architecture (hidden layers, dropout)
- Privacy parameters (experimental)
- MITRE ATT&CK security rules

## Testing Different Scenarios

### Without LLM (Faster)
```bash
python client.py --client_id 1 --no-llm
```

### With OpenAI Explanations
1. Set API key: `export OPENAI_API_KEY="your-key"`
2. Run: `python client.py --client_id 1`

### Experimental Privacy Mode
```bash
python client.py --client_id 1 --enable-privacy
```
⚠️ Warning: Currently incompatible with GNN layers, will auto-disable

## Expected Behavior

### Successful Run
- Server waits for clients
- Clients connect and download initial model
- 3 rounds of training occur
- Each round: fit → aggregate → evaluate
- Clients generate security alerts for detected anomalies
- Server completes all rounds successfully

### Common Issues

**"ModuleNotFoundError: xxhash"**
```bash
pip install xxhash aiohttp psutil pyparsing
```

**"Server not responding"**
- Check server is running on port 8080
- Verify no firewall blocking localhost

**"Privacy errors"**
- DP is experimental, use `--enable-privacy` only for testing
- Default behavior (no flag) disables DP automatically

## Performance

- **Training time**: ~2-3 seconds per round with 2 clients
- **Total demo time**: ~10-15 seconds for 3 rounds
- **Memory usage**: ~200MB per client (CPU-only PyTorch)
- **Network**: Localhost only (no external connections except LLM API)

## Next Steps

1. **Custom Data**: Modify `data_loader.py` to load real network traffic
2. **More Clients**: Run additional clients with different `--client_id` values
3. **More Rounds**: Adjust `--rounds` parameter on server
4. **Production Setup**: Implement TLS encryption and custom DP grad samplers
