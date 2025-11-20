import argparse
import flwr as fl
from flwr.server.strategy import FedAvg


def main():
    parser = argparse.ArgumentParser(description="FedSymGraph Federated Server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FedSymGraph Global Coordinator")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Federated Rounds: {args.rounds}")
    print(f"Minimum Clients Required: {args.min_clients}")
    print("="*60 + "\n")
    print("Waiting for federated clients to connect...")
    print(f"Clients should connect to: 127.0.0.1:{args.port}\n")
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*60)
    print("Federated Learning Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
