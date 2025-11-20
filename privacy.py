import torch
import numpy as np

class DifferentialPrivacyWrapper:
    def __init__(self, model, optimizer, data_loader, noise_multiplier=1.1, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.steps = 0
    
    def clip_gradients(self):
        params = [p for p in self.model.parameters() if p.grad is not None]
        total_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in params))
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                p.grad.mul_(clip_coef)
    
    def add_noise(self):
        noise_scale = self.noise_multiplier * self.max_grad_norm
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(0, noise_scale, size=p.grad.shape, device=p.device)
                p.grad.add_(noise)
    
    def step(self):
        self.clip_gradients()
        self.add_noise()
        self.optimizer.step()
        self.steps += 1
    
    def get_epsilon(self, delta=1e-5, num_clients=2):
        if self.steps == 0:
            return 0.0
        c = np.sqrt(self.steps * np.log(1 / delta))
        return c / self.noise_multiplier
    
    def get_model(self):
        return self.model
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_data_loader(self):
        return self.data_loader
