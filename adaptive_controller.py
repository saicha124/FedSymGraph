"""
Adaptive Reinforcement Controller for FedSymGraph
Dynamically adjusts detection thresholds, privacy levels, and communication intervals.
Based on Q-learning with epsilon-greedy exploration.
"""

import torch
import numpy as np
from collections import deque
import json


class AdaptiveController:
    """
    Reinforcement Learning controller that adapts:
    1. Anomaly detection threshold
    2. Privacy noise level (epsilon)
    3. Communication interval (rounds between updates)
    
    Uses Q-learning with discretized state/action spaces.
    """
    
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # State space: (accuracy_level, false_positive_rate, privacy_budget_used)
        # Each dimension has 5 discrete levels (Low, Medium-Low, Medium, Medium-High, High)
        self.state_dims = (5, 5, 5)
        
        # Action space: (threshold_adjustment, privacy_adjustment, comm_adjustment)
        # threshold_adjustment: -2 (decrease a lot), -1 (decrease), 0 (keep), +1 (increase), +2 (increase a lot)
        # privacy_adjustment: -1 (less noise), 0 (keep), +1 (more noise)
        # comm_adjustment: -1 (more frequent), 0 (keep), +1 (less frequent)
        self.threshold_actions = [-0.1, -0.05, 0.0, 0.05, 0.1]
        self.privacy_actions = [-0.5, 0.0, 0.5]
        self.comm_actions = [-1, 0, 1]
        
        # Q-table: state -> action -> value
        self.q_table = {}
        
        # Current parameters
        self.detection_threshold = 0.5
        self.privacy_noise = 1.0  # noise multiplier
        self.comm_interval = 1  # rounds between updates
        
        # History for tracking
        self.history = {
            'rewards': [],
            'thresholds': [],
            'privacy_levels': [],
            'accuracies': [],
            'false_positives': []
        }
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=1000)
    
    def discretize_state(self, accuracy, false_positive_rate, privacy_used):
        """
        Convert continuous metrics to discrete state.
        
        Args:
            accuracy: Detection accuracy (0-1)
            false_positive_rate: FP rate (0-1)
            privacy_used: Privacy budget used (0-1, normalized)
        
        Returns:
            Tuple representing discrete state
        """
        # Discretize to 5 levels each
        acc_level = min(int(accuracy * 5), 4)
        fp_level = min(int(false_positive_rate * 5), 4)
        privacy_level = min(int(privacy_used * 5), 4)
        
        return (acc_level, fp_level, privacy_level)
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Returns:
            Tuple of (threshold_idx, privacy_idx, comm_idx)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            threshold_idx = np.random.randint(0, len(self.threshold_actions))
            privacy_idx = np.random.randint(0, len(self.privacy_actions))
            comm_idx = np.random.randint(0, len(self.comm_actions))
        else:
            # Exploit: best action
            best_value = float('-inf')
            best_action = (2, 1, 1)  # default: no change
            
            for t_idx in range(len(self.threshold_actions)):
                for p_idx in range(len(self.privacy_actions)):
                    for c_idx in range(len(self.comm_actions)):
                        action = (t_idx, p_idx, c_idx)
                        q_value = self.get_q_value(state, action)
                        
                        if q_value > best_value:
                            best_value = q_value
                            best_action = action
            
            threshold_idx, privacy_idx, comm_idx = best_action
        
        return threshold_idx, privacy_idx, comm_idx
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule."""
        current_q = self.get_q_value(state, action)
        
        # Find max Q-value for next state
        max_next_q = float('-inf')
        for t_idx in range(len(self.threshold_actions)):
            for p_idx in range(len(self.privacy_actions)):
                for c_idx in range(len(self.comm_actions)):
                    next_action = (t_idx, p_idx, c_idx)
                    next_q = self.get_q_value(next_state, next_action)
                    max_next_q = max(max_next_q, next_q)
        
        if max_next_q == float('-inf'):
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
    
    def compute_reward(self, accuracy, false_positive_rate, privacy_used):
        """
        Compute reward based on detection performance and privacy preservation.
        
        Reward function balances:
        - High accuracy (positive)
        - Low false positives (positive)
        - Privacy preservation (bonus for using less budget)
        """
        # Accuracy reward (0 to 1)
        accuracy_reward = accuracy
        
        # False positive penalty (-1 to 0)
        fp_penalty = -false_positive_rate
        
        # Privacy bonus (0 to 0.5) - reward for using less privacy budget
        privacy_bonus = 0.5 * (1.0 - privacy_used)
        
        # Combined reward
        reward = accuracy_reward + fp_penalty + privacy_bonus
        
        return reward
    
    def adapt(self, accuracy, false_positive_rate, privacy_used):
        """
        Adapt parameters based on current performance.
        
        Args:
            accuracy: Current detection accuracy (0-1)
            false_positive_rate: Current FP rate (0-1)
            privacy_used: Privacy budget used (0-1, normalized)
        
        Returns:
            Dict with updated parameters
        """
        # Get current state
        state = self.discretize_state(accuracy, false_positive_rate, privacy_used)
        
        # Choose action
        threshold_idx, privacy_idx, comm_idx = self.choose_action(state)
        
        # Apply actions
        threshold_adjustment = self.threshold_actions[threshold_idx]
        privacy_adjustment = self.privacy_actions[privacy_idx]
        comm_adjustment = self.comm_actions[comm_idx]
        
        # Update parameters
        self.detection_threshold = np.clip(
            self.detection_threshold + threshold_adjustment,
            0.1, 0.9
        )
        
        self.privacy_noise = np.clip(
            self.privacy_noise + privacy_adjustment,
            0.5, 3.0
        )
        
        self.comm_interval = max(1, self.comm_interval + comm_adjustment)
        
        # Compute reward
        reward = self.compute_reward(accuracy, false_positive_rate, privacy_used)
        
        # Store experience
        action = (threshold_idx, privacy_idx, comm_idx)
        self.replay_buffer.append((state, action, reward))
        
        # Update Q-table (if we have next state)
        if len(self.replay_buffer) > 1:
            prev_state, prev_action, prev_reward = self.replay_buffer[-2]
            self.update_q_table(prev_state, prev_action, prev_reward, state)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update history
        self.history['rewards'].append(reward)
        self.history['thresholds'].append(self.detection_threshold)
        self.history['privacy_levels'].append(self.privacy_noise)
        self.history['accuracies'].append(accuracy)
        self.history['false_positives'].append(false_positive_rate)
        
        return {
            'detection_threshold': self.detection_threshold,
            'privacy_noise': self.privacy_noise,
            'comm_interval': self.comm_interval,
            'reward': reward,
            'epsilon': self.epsilon
        }
    
    def get_parameters(self):
        """Get current adaptive parameters."""
        return {
            'detection_threshold': self.detection_threshold,
            'privacy_noise': self.privacy_noise,
            'comm_interval': self.comm_interval
        }
    
    def save_policy(self, path="adaptive_policy.json"):
        """Save learned Q-table to file."""
        # Convert tuple keys to strings for JSON
        q_table_str = {}
        for state, actions in self.q_table.items():
            state_str = str(state)
            q_table_str[state_str] = {}
            for action, value in actions.items():
                action_str = str(action)
                q_table_str[state_str][action_str] = value
        
        policy_data = {
            'q_table': q_table_str,
            'parameters': self.get_parameters(),
            'history': self.history
        }
        
        with open(path, 'w') as f:
            json.dump(policy_data, f, indent=2)
        
        print(f"✓ Saved adaptive policy to {path}")
    
    def load_policy(self, path="adaptive_policy.json"):
        """Load learned Q-table from file."""
        try:
            with open(path, 'r') as f:
                policy_data = json.load(f)
            
            # Convert string keys back to tuples
            self.q_table = {}
            for state_str, actions in policy_data['q_table'].items():
                state = eval(state_str)
                self.q_table[state] = {}
                for action_str, value in actions.items():
                    action = eval(action_str)
                    self.q_table[state][action] = value
            
            # Load parameters
            params = policy_data['parameters']
            self.detection_threshold = params['detection_threshold']
            self.privacy_noise = params['privacy_noise']
            self.comm_interval = params['comm_interval']
            
            print(f"✓ Loaded adaptive policy from {path}")
            return True
        except FileNotFoundError:
            print(f"Policy file {path} not found. Starting with fresh Q-table.")
            return False
    
    def get_stats(self):
        """Get statistics about controller performance."""
        if not self.history['rewards']:
            return "No history yet."
        
        stats = {
            'avg_reward': np.mean(self.history['rewards']),
            'avg_accuracy': np.mean(self.history['accuracies']),
            'avg_fp_rate': np.mean(self.history['false_positives']),
            'current_threshold': self.detection_threshold,
            'current_privacy': self.privacy_noise,
            'episodes': len(self.history['rewards']),
            'q_table_size': len(self.q_table)
        }
        
        return stats


# Test script
if __name__ == "__main__":
    print("Testing Adaptive Controller...")
    
    controller = AdaptiveController()
    
    # Simulate multiple rounds of training
    print("\nSimulating 10 training rounds:")
    for round_num in range(10):
        # Simulate varying performance metrics
        accuracy = 0.7 + np.random.random() * 0.2  # 0.7 - 0.9
        fp_rate = 0.1 + np.random.random() * 0.1  # 0.1 - 0.2
        privacy_used = 0.3 + np.random.random() * 0.3  # 0.3 - 0.6
        
        # Adapt parameters
        result = controller.adapt(accuracy, fp_rate, privacy_used)
        
        print(f"Round {round_num + 1}:")
        print(f"  Threshold: {result['detection_threshold']:.3f}")
        print(f"  Privacy Noise: {result['privacy_noise']:.2f}")
        print(f"  Reward: {result['reward']:.3f}")
        print(f"  Epsilon: {result['epsilon']:.3f}")
    
    # Show statistics
    print("\nController Statistics:")
    stats = controller.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save policy
    controller.save_policy("test_policy.json")
    
    print("\n✓ Adaptive controller working correctly!")
