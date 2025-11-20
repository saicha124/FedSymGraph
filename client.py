import argparse
import flwr as fl
import torch
from model import FedGNN
from data_loader import load_local_graphs
from reasoning import HybridExplainer
from privacy import DifferentialPrivacyWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FedSymClient(fl.client.NumPyClient):
    """
    Federated Learning client with privacy-preserving training and
    neuro-symbolic explainability.
    """
    def __init__(self, client_id, use_privacy=True, use_openai=True):
        self.client_id = client_id
        self.use_privacy = use_privacy
        
        self.train_loader, self.test_loader = load_local_graphs(client_id)
        self.model = FedGNN(num_features=12, hidden_channels=32, num_classes=2).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.explainer = HybridExplainer(use_openai=use_openai)
        
        if self.use_privacy:
            try:
                privacy_wrapper = DifferentialPrivacyWrapper(
                    model=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=1.1,
                    max_grad_norm=1.0
                )
                self.model = privacy_wrapper.get_model()
                self.optimizer = privacy_wrapper.get_optimizer()
                self.train_loader = privacy_wrapper.get_data_loader()
                print(f"[Client {self.client_id}] Differential Privacy enabled")
            except Exception as e:
                print(f"[Client {self.client_id}] Privacy setup failed: {e}. Running without DP.")
                self.use_privacy = False

    def fit(self, parameters, config):
        """Train the model locally on client data."""
        self.set_parameters(parameters)
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(1):
            for batch in self.train_loader:
                batch = batch.to(DEVICE)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.CrossEntropyLoss()(out, batch.y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[Client {self.client_id}] Training loss: {avg_loss:.4f}")
        
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        """Evaluate model and generate explanations for detected anomalies."""
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        alerts_generated = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(DEVICE)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.CrossEntropyLoss()(out, batch.y)
                pred = out.argmax(dim=1)
                
                total_loss += loss.item()
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
                if 1 in pred and alerts_generated < 2:
                    mock_features = {
                        'auth_fail': 12,
                        'protocol': 'SSH',
                        'bytes_out': 500000,
                        'dst_port': 22,
                        'service': 'SSH',
                        'file_access': 'read'
                    }
                    
                    tactics = self.explainer.symbolic_inference(mock_features)
                    confidence = torch.softmax(out[pred == 1][0], dim=0)[1].item()
                    report = self.explainer.generate_explanation(tactics, confidence)
                    
                    print(f"\n[Client {self.client_id} ALERT]: {report}\n")
                    alerts_generated += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(avg_loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def get_parameters(self, config=None):
        """Extract model weights as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load model weights from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)


def main():
    parser = argparse.ArgumentParser(description="FedSymGraph Federated Client")
    parser.add_argument("--client_id", type=int, default=1, help="Client ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--no-privacy", action="store_true", help="Disable differential privacy")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM explanations")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FedSymGraph Client {args.client_id} Starting")
    print(f"Server: {args.server}")
    print(f"Privacy: {'Disabled' if args.no_privacy else 'Enabled'}")
    print(f"LLM Explanations: {'Disabled' if args.no_llm else 'Enabled'}")
    print(f"{'='*60}\n")
    
    client = FedSymClient(
        client_id=args.client_id,
        use_privacy=not args.no_privacy,
        use_openai=not args.no_llm
    )
    
    fl.client.start_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()
