"""
Configuration file for FedSymGraph
"""

GNN_CONFIG = {
    "num_features": 12,
    "hidden_channels": 32,
    "num_classes": 2,
    "dropout": 0.5,
    "learning_rate": 0.01
}

PRIVACY_CONFIG = {
    "enabled": True,
    "noise_multiplier": 1.1,
    "max_grad_norm": 1.0,
    "target_delta": 1e-5
}

FEDERATED_CONFIG = {
    "num_rounds": 5,
    "min_clients": 2,
    "server_port": 8080,
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0
}

DATA_CONFIG = {
    "batch_size": 8,
    "num_graphs_per_client": 100,
    "test_split": 0.2,
    "num_nodes_per_graph": 20
}

LLM_CONFIG = {
    "use_openai": True,
    "model": "gpt-3.5-turbo",
    "max_tokens": 100,
    "temperature": 0.7
}

MITRE_RULES = [
    {
        "rule": "auth_fail > 10 and protocol == 'SSH'",
        "tactic": "T1110: Brute Force",
        "description": "Multiple failed authentication attempts detected"
    },
    {
        "rule": "bytes_out > 1000000 and dst_port == 443",
        "tactic": "T1048: Exfiltration",
        "description": "Large data transfer over HTTPS detected"
    },
    {
        "rule": "service == 'SMB' and file_access == 'write'",
        "tactic": "T1021: Lateral Movement",
        "description": "SMB file write operation detected"
    }
]
