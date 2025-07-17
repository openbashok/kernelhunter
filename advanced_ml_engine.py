#!/usr/bin/env python3
"""
Advanced Machine Learning Engine for KernelHunter
Implements DQN, Policy Gradient, and Transformer models for intelligent shellcode generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import math
from dataclasses import dataclass
import asyncio
import aiohttp
from pathlib import Path

# Experience replay buffer for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class MLConfig:
    """Configuration for ML models"""
    dqn_hidden_size: int = 256
    dqn_learning_rate: float = 0.001
    dqn_gamma: float = 0.99
    dqn_epsilon: float = 0.1
    dqn_buffer_size: int = 10000
    dqn_batch_size: int = 32
    
    policy_lr: float = 0.001
    policy_hidden_size: int = 128
    
    transformer_embed_dim: int = 512
    transformer_nhead: int = 8
    transformer_nlayers: int = 6
    transformer_dropout: float = 0.1
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNNetwork(nn.Module):
    """Deep Q-Network for attack strategy selection"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    """Policy Network for mutation strategy optimization"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ShellcodeTransformer(nn.Module):
    """Transformer model for intelligent shellcode generation"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, nhead: int = 8, 
                 nlayers: int = 6, max_len: int = 1024):
        super(ShellcodeTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Output projection
        return self.output_projection(x)

class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class ShellcodeDataset(Dataset):
    """Dataset for training the transformer model"""
    
    def __init__(self, shellcodes: List[bytes], max_len: int = 1024):
        self.shellcodes = shellcodes
        self.max_len = max_len
        self.vocab_size = 256  # Byte vocabulary
        
    def __len__(self):
        return len(self.shellcodes)
        
    def __getitem__(self, idx):
        shellcode = self.shellcodes[idx]
        
        # Convert to tensor and pad/truncate
        tensor = torch.tensor(list(shellcode), dtype=torch.long)
        if len(tensor) > self.max_len:
            tensor = tensor[:self.max_len]
        else:
            padding = torch.zeros(self.max_len - len(tensor), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
            
        return tensor[:-1], tensor[1:]  # Input, target

class AdvancedMLEngine:
    """Advanced Machine Learning Engine for KernelHunter"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.dqn_network = DQNNetwork(64, 50).to(self.device)  # 50 attack types
        self.dqn_target = DQNNetwork(64, 50).to(self.device)
        self.dqn_optimizer = optim.Adam(self.dqn_network.parameters(), lr=config.dqn_learning_rate)
        
        self.policy_network = PolicyNetwork(64, 10).to(self.device)  # 10 mutation types
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=config.policy_lr)
        
        self.transformer = ShellcodeTransformer(256).to(self.device)
        self.transformer_optimizer = optim.Adam(self.transformer.parameters(), lr=0.0001)
        
        # Initialize replay buffer
        self.replay_buffer = ExperienceReplayBuffer(config.dqn_buffer_size)
        
        # Training state
        self.dqn_losses = []
        self.policy_losses = []
        self.transformer_losses = []
        
        # Load pre-trained models if available
        self.load_models()
        
    def extract_state_features(self, shellcode: bytes, crash_info: Dict) -> torch.Tensor:
        """Extract features from shellcode and crash info for ML models"""
        features = []
        
        # Shellcode features
        features.extend([
            len(shellcode),  # Length
            sum(shellcode) / len(shellcode) if shellcode else 0,  # Average byte value
            len(set(shellcode)) / len(shellcode) if shellcode else 0,  # Entropy
            shellcode.count(b'\x90'),  # NOP count
            shellcode.count(b'\x0f'),  # Extended opcode count
            shellcode.count(b'\x48'),  # REX prefix count
        ])
        
        # Crash features
        features.extend([
            crash_info.get('crash_rate', 0),
            crash_info.get('system_impact', 0),
            crash_info.get('generation', 0),
            crash_info.get('population_size', 0),
        ])
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0)
        features = features[:64]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def select_attack_strategy(self, state: torch.Tensor, epsilon: float = None) -> int:
        """Select attack strategy using DQN with epsilon-greedy exploration"""
        if epsilon is None:
            epsilon = self.config.dqn_epsilon
            
        if random.random() < epsilon:
            return random.randint(0, 49)  # Random attack
        
        with torch.no_grad():
            q_values = self.dqn_network(state.unsqueeze(0))
            return q_values.argmax().item()
    
    def select_mutation_strategy(self, state: torch.Tensor) -> int:
        """Select mutation strategy using policy network"""
        with torch.no_grad():
            probs = self.policy_network(state.unsqueeze(0))
            return torch.multinomial(probs, 1).item()
    
    def generate_shellcode_transformer(self, seed: bytes, max_length: int = 64) -> bytes:
        """Generate shellcode using transformer model"""
        self.transformer.eval()
        
        # Convert seed to tensor
        seed_tensor = torch.tensor(list(seed), dtype=torch.long, device=self.device)
        if len(seed_tensor) > self.config.transformer_embed_dim:
            seed_tensor = seed_tensor[:self.config.transformer_embed_dim]
        
        generated = seed_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_length - len(seed)):
                # Get model prediction
                input_seq = generated.unsqueeze(0)
                output = self.transformer(input_seq)
                
                # Sample next byte
                logits = output[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_byte = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_byte])
                
                # Stop if we generate too many bytes
                if len(generated) >= max_length:
                    break
        
        return bytes(generated.cpu().numpy())
    
    def train_dqn(self, batch_size: int = None):
        """Train DQN network using experience replay"""
        if batch_size is None:
            batch_size = self.config.dqn_batch_size
            
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states = torch.stack([exp.state for exp in batch])
        actions = torch.tensor([exp.action for exp in batch], device=self.device)
        rewards = torch.tensor([exp.reward for exp in batch], device=self.device)
        next_states = torch.stack([exp.next_state for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], device=self.device)
        
        # Compute Q values
        current_q_values = self.dqn_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.dqn_target(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config.dqn_gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        
        self.dqn_losses.append(loss.item())
        
        # Update target network periodically
        if len(self.dqn_losses) % 100 == 0:
            self.dqn_target.load_state_dict(self.dqn_network.state_dict())
    
    def train_policy(self, states: List[torch.Tensor], actions: List[int], 
                    rewards: List[float]):
        """Train policy network using policy gradient"""
        if not states:
            return
        
        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get action probabilities
        probs = self.policy_network(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = -(torch.log(action_probs) * rewards).mean()
        
        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        self.policy_losses.append(loss.item())
    
    def train_transformer(self, shellcodes: List[bytes], epochs: int = 5):
        """Train transformer model on shellcode data"""
        if not shellcodes:
            return
        
        dataset = ShellcodeDataset(shellcodes)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.transformer.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.transformer(inputs)
                loss = F.cross_entropy(outputs.view(-1, 256), targets.view(-1))
                
                # Backward pass
                self.transformer_optimizer.zero_grad()
                loss.backward()
                self.transformer_optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Transformer Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            self.transformer_losses.append(avg_loss)
            print(f"Transformer Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    
    def update_experience(self, state: torch.Tensor, action: int, reward: float, 
                         next_state: torch.Tensor, done: bool):
        """Add experience to replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def predict_crash_probability(self, shellcode: bytes) -> float:
        """Predict probability of crash for a given shellcode"""
        # Simple heuristic-based prediction
        features = self.extract_state_features(shellcode, {})
        
        # Use a simple neural network for prediction
        with torch.no_grad():
            # This is a simplified version - in practice you'd train a dedicated model
            crash_prob = torch.sigmoid(features.mean()).item()
            return crash_prob
    
    def get_anomaly_score(self, shellcode: bytes, crash_info: Dict) -> float:
        """Calculate anomaly score for shellcode"""
        features = self.extract_state_features(shellcode, crash_info)
        
        # Simple anomaly detection based on feature statistics
        # In practice, you'd use more sophisticated methods like Isolation Forest
        anomaly_score = torch.std(features).item()
        return anomaly_score
    
    def save_models(self, path: str = "./ml_models"):
        """Save trained models"""
        Path(path).mkdir(exist_ok=True)
        
        torch.save(self.dqn_network.state_dict(), f"{path}/dqn_network.pth")
        torch.save(self.policy_network.state_dict(), f"{path}/policy_network.pth")
        torch.save(self.transformer.state_dict(), f"{path}/transformer.pth")
        
        # Save training history
        history = {
            'dqn_losses': self.dqn_losses,
            'policy_losses': self.policy_losses,
            'transformer_losses': self.transformer_losses
        }
        
        with open(f"{path}/training_history.json", 'w') as f:
            json.dump(history, f)
    
    def load_models(self, path: str = "./ml_models"):
        """Load trained models"""
        try:
            self.dqn_network.load_state_dict(torch.load(f"{path}/dqn_network.pth", 
                                                       map_location=self.device))
            self.policy_network.load_state_dict(torch.load(f"{path}/policy_network.pth", 
                                                          map_location=self.device))
            self.transformer.load_state_dict(torch.load(f"{path}/transformer.pth", 
                                                       map_location=self.device))
            
            # Load training history
            with open(f"{path}/training_history.json", 'r') as f:
                history = json.load(f)
                self.dqn_losses = history.get('dqn_losses', [])
                self.policy_losses = history.get('policy_losses', [])
                self.transformer_losses = history.get('transformer_losses', [])
                
            print("✅ ML models loaded successfully")
        except FileNotFoundError:
            print("⚠️ No pre-trained models found, starting fresh")
    
    def get_model_stats(self) -> Dict:
        """Get statistics about model performance"""
        return {
            'dqn_losses': len(self.dqn_losses),
            'policy_losses': len(self.policy_losses),
            'transformer_losses': len(self.transformer_losses),
            'replay_buffer_size': len(self.replay_buffer),
            'avg_dqn_loss': np.mean(self.dqn_losses[-100:]) if self.dqn_losses else 0,
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_transformer_loss': np.mean(self.transformer_losses[-100:]) if self.transformer_losses else 0
        }

# Global ML engine instance
ml_engine = None

def get_ml_engine() -> AdvancedMLEngine:
    """Get or create global ML engine instance"""
    global ml_engine
    if ml_engine is None:
        config = MLConfig()
        ml_engine = AdvancedMLEngine(config)
    return ml_engine

if __name__ == "__main__":
    # Test the ML engine
    engine = get_ml_engine()
    
    # Test shellcode generation
    test_shellcode = b"\x90\x90\x48\x31\xc0\x0f\x05"
    generated = engine.generate_shellcode_transformer(test_shellcode)
    print(f"Generated shellcode: {generated.hex()}")
    
    # Test state extraction
    state = engine.extract_state_features(test_shellcode, {'crash_rate': 0.1})
    print(f"State features shape: {state.shape}")
    
    # Test attack selection
    attack = engine.select_attack_strategy(state)
    print(f"Selected attack: {attack}")
    
    # Test mutation selection
    mutation = engine.select_mutation_strategy(state)
    print(f"Selected mutation: {mutation}") 