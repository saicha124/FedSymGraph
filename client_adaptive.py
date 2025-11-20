"""
Enhanced Federated Client with Integrated Adaptive Controller
Integrates the Q-learning adaptive controller into the federated training loop.
"""

import argparse
import flwr as fl
import torch
from model_heterogeneous import FedHeteroGNN
from data_loader_heterogeneous import load_heterogeneous_for_client
from reasoning import HybridExplainer
from privacy import DifferentialPrivacyWrapper
from adaptive_controller import AdaptiveController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveFedSymClient(fl.client.NumPyClient):
    """
    Federated Learning client with:
    - Heterogeneous GNN
    - Adaptive Q-learning controller
    - Privacy preservation
    - Neuro-symbolic explainability
    """
    
    def __init__(self, client_id, use_privacy=False, use_openai=True, use_adaptive=True):
        self.client_id = client_id
        self.use_privacy = use_privacy
        self.use_adaptive = use_adaptive
        
        # Load heterogeneous graphs with balanced labels
        print(f"[Client {self.client_id}] Loading heterogeneous graphs...")
        self.train_loader, self.test_loader = load_heterogeneous_for_client(
            client_id=client_id,
            csv_path="TON_IoT_Network.csv",
            batch_size=8,
            flows_per_graph=50,
            balanced=True  # Ensures balanced normal/attack distribution
        )
        
        # Heterogeneous GNN model
        self.model = FedHeteroGNN(
            hidden_channels=64,
            num_classes=2,
            num_layers=2
        ).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.explainer = HybridExplainer(use_openai=use_openai)
        
        # Adaptive controller
        if self.use_adaptive:
            self.controller = AdaptiveController(
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=1.0,
                epsilon_decay=0.995
            )
            print(f"[Client {self.client_id}] Adaptive controller ENABLED")
            
            # Try to load existing policy
            self.controller.load_policy(f"client_{client_id}_policy.json")
        else:
            self.controller = None
        
        # Detection threshold (adapted by controller)
        self.detection_threshold = 0.5
        
        # Privacy wrapper
        self.privacy_wrapper = None
        if self.use_privacy:
            print(f"[Client {self.client_id}] Differential Privacy ENABLED")
            self.privacy_wrapper = DifferentialPrivacyWrapper(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0
            )
        
        # Performance tracking
        self.round_num = 0
        self.accuracy_history = []
        self.fp_rate_history = []
    
    def fit(self, parameters, config):
        """Train the model with adaptive parameter adjustment."""
        self.set_parameters(parameters)
        self.round_num += 1
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Apply adaptive privacy noise if controller is enabled
        if self.use_adaptive and self.controller and self.use_privacy and self.privacy_wrapper:
            params = self.controller.get_parameters()
            self.privacy_wrapper.noise_multiplier = params['privacy_noise']
            print(f"[Client {self.client_id}] Using adaptive privacy noise: {params['privacy_noise']:.2f}")
        
        for epoch in range(1):
            for batch in self.train_loader:
                batch = batch.to(DEVICE)
                self.optimizer.zero_grad()
                # Forward pass with HeteroData batch (FedHeteroGNN handles unpacking internally)
                out = self.model(batch)
                loss = torch.nn.CrossEntropyLoss()(out, batch.y)
                loss.backward()
                
                if self.use_privacy and self.privacy_wrapper:
                    self.privacy_wrapper.step()
                else:
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        if self.use_privacy and self.privacy_wrapper:
            epsilon = self.privacy_wrapper.get_epsilon(delta=1e-5)
            print(f"[Client {self.client_id}] Round {self.round_num} - Training loss: {avg_loss:.4f}, Privacy: ε={epsilon:.2f}")
        else:
            print(f"[Client {self.client_id}] Round {self.round_num} - Training loss: {avg_loss:.4f}")
        
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": avg_loss}
    
    def evaluate(self, parameters, config):
        """Evaluate model with adaptive threshold and update controller."""
        self.set_parameters(parameters)
        self.model.eval()
        
        # Apply adaptive detection threshold if controller is enabled
        if self.use_adaptive and self.controller:
            params = self.controller.get_parameters()
            self.detection_threshold = params['detection_threshold']
            print(f"[Client {self.client_id}] Using adaptive threshold: {self.detection_threshold:.3f}")
        
        total_loss = 0
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        alerts_generated = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(DEVICE)
                out = self.model(batch)
                loss = torch.nn.CrossEntropyLoss()(out, batch.y)
                
                # Apply adaptive threshold
                probs = torch.softmax(out, dim=1)
                pred = (probs[:, 1] > self.detection_threshold).long()
                
                total_loss += loss.item()
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
                # Track TP, FP, TN, FN for FP rate
                for p, t in zip(pred, batch.y):
                    if p == 1 and t == 1:
                        true_positives += 1
                    elif p == 1 and t == 0:
                        false_positives += 1
                    elif p == 0 and t == 0:
                        true_negatives += 1
                    else:
                        false_negatives += 1
                
                # Generate explanations for detected attacks
                if 1 in pred and alerts_generated < 2:
                    attack_indices = (pred == 1).nonzero(as_tuple=True)[0]
                    for idx in attack_indices[:1]:  # Show one alert
                        mock_features = {
                            'auth_fail': 12,
                            'protocol': 'SSH',
                            'bytes_out': 500000,
                            'dst_port': 22,
                            'service': 'SSH',
                            'file_access': 'read'
                        }
                        
                        tactics = self.explainer.symbolic_inference(mock_features)
                        confidence = probs[idx, 1].item()
                        report = self.explainer.generate_explanation(tactics, confidence)
                        
                        print(f"\n[Client {self.client_id} ALERT]: {report}\n")
                        alerts_generated += 1
                        break
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        
        # Calculate false positive rate
        fp_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Calculate privacy budget used (normalized to 0-1)
        privacy_used = 0.0
        if self.use_privacy and self.privacy_wrapper:
            epsilon = self.privacy_wrapper.get_epsilon(delta=1e-5)
            privacy_used = min(epsilon / 10.0, 1.0)  # Normalize assuming max epsilon of 10
        
        # Update adaptive controller
        if self.use_adaptive and self.controller:
            result = self.controller.adapt(accuracy, fp_rate, privacy_used)
            print(f"[Client {self.client_id}] Controller adapted:")
            print(f"  Threshold: {result['detection_threshold']:.3f}")
            print(f"  Privacy Noise: {result['privacy_noise']:.2f}")
            print(f"  Reward: {result['reward']:.3f}")
            
            # Save policy periodically
            if self.round_num % 5 == 0:
                self.controller.save_policy(f"client_{self.client_id}_policy.json")
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, FP Rate: {fp_rate:.4f}")
        
        # Store history
        self.accuracy_history.append(accuracy)
        self.fp_rate_history.append(fp_rate)
        
        return float(avg_loss), len(self.test_loader.dataset), {
            "accuracy": float(accuracy),
            "fp_rate": float(fp_rate),
            "true_positives": true_positives,
            "false_positives": false_positives
        }
    
    def get_parameters(self, config=None):
        """Extract model weights as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Load model weights from NumPy arrays with correct dtype."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {}
        for k, v in params_dict:
            # Get the expected dtype from the model parameter
            param = self.model.state_dict()[k]
            # Convert numpy array to tensor with correct dtype
            state_dict[k] = torch.from_numpy(v).to(param.dtype)
        self.model.load_state_dict(state_dict, strict=False)


def main():
    """Run adaptive federated client."""
    parser = argparse.ArgumentParser(description="Adaptive FedSymGraph Client")
    parser.add_argument("--client_id", type=int, default=1, help="Client ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--enable-privacy", action="store_true", help="Enable differential privacy")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM explanations")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive controller")
    args = parser.parse_args()
    
    print(f"""
============================================================
Adaptive FedSymGraph Client {args.client_id}
============================================================
Features:
  ✓ Heterogeneous GNN (IP, Port, Protocol nodes)
  ✓ Balanced graph sampling (normal/attack)
  ✓ Adaptive Q-learning controller
  {'✓' if args.enable_privacy else '✗'} Differential Privacy
  {'✓' if not args.no_llm else '✗'} LLM Explanations
  {'✓' if not args.no_adaptive else '✗'} Adaptive Optimization
============================================================
""")
    
    # Create adaptive client
    client = AdaptiveFedSymClient(
        client_id=args.client_id,
        use_privacy=args.enable_privacy,
        use_openai=not args.no_llm,
        use_adaptive=not args.no_adaptive
    )
    
    # Start federated learning
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client()
    )


if __name__ == "__main__":
    main()
