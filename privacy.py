import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class DifferentialPrivacyWrapper:
    """
    Differential Privacy wrapper for federated clients.
    Ensures that gradients don't leak sensitive information about local data.
    """
    def __init__(self, model, optimizer, data_loader, noise_multiplier=1.1, max_grad_norm=1.0):
        self.privacy_engine = PrivacyEngine()
        
        model = ModuleValidator.fix(model)
        
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    
    def get_epsilon(self, delta=1e-5):
        """Calculate privacy budget (epsilon) spent so far."""
        return self.privacy_engine.get_epsilon(delta)
    
    def get_model(self):
        return self.model
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_data_loader(self):
        return self.data_loader
