#!/usr/bin/env python3
"""
Integration test for FedSymGraph
Tests that the federated learning loop runs successfully
"""

import multiprocessing
import time
import sys
import flwr as fl
from client import FedSymClient


def run_server(num_rounds=2, min_clients=2, port=8082):
    """Run federated server in a subprocess"""
    try:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
        )
        
        print(f"[TEST SERVER] Starting on port {port}...")
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        print("[TEST SERVER] Completed successfully")
    except Exception as e:
        print(f"[TEST SERVER] Error: {e}")
        sys.exit(1)


def run_client(client_id, server_address):
    """Run federated client in a subprocess"""
    try:
        print(f"[TEST CLIENT {client_id}] Starting...")
        client = FedSymClient(
            client_id=client_id,
            use_privacy=False,
            use_openai=False
        )
        
        fl.client.start_client(
            server_address=server_address,
            client=client
        )
        print(f"[TEST CLIENT {client_id}] Completed successfully")
    except Exception as e:
        print(f"[TEST CLIENT {client_id}] Error: {e}")
        sys.exit(1)


def main():
    print("\n" + "="*60)
    print("FedSymGraph Integration Test")
    print("="*60)
    print("Testing federated learning with 2 clients, 2 rounds\n")
    
    port = 8082
    server_address = f"127.0.0.1:{port}"
    
    server_process = multiprocessing.Process(
        target=run_server,
        args=(2, 2, port)
    )
    server_process.start()
    
    time.sleep(3)
    
    client_processes = []
    for client_id in [1, 2]:
        p = multiprocessing.Process(
            target=run_client,
            args=(client_id, server_address)
        )
        p.start()
        client_processes.append(p)
        time.sleep(1)
    
    for p in client_processes:
        p.join(timeout=60)
        if p.exitcode != 0:
            print(f"\n[TEST] FAILED: Client process exited with code {p.exitcode}")
            server_process.terminate()
            sys.exit(1)
    
    server_process.join(timeout=10)
    server_process.terminate()
    
    print("\n" + "="*60)
    print("[TEST] SUCCESS: Federated learning completed")
    print("="*60 + "\n")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    sys.exit(main())
