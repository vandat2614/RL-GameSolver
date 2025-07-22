import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

class Logger:
    def __init__(self, config: dict):

        log_dir = config.get("logger", {}).get("log_dir", "runs")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create weights directory
        self.weights_dir = self.log_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.log_dir / 'config.yaml', 'w') as file:
            json.dump(config, file, indent=4)
        
        self.rewards = []
        self.losses = []
        self.epsilons = []
        
    def log_episode(self, episode: int, reward: float, loss: float, epsilon: float):
        self.rewards.append(reward)
        self.losses.append(loss)
        self.epsilons.append(epsilon)

        # Print progress
        print(f"Episode {episode}, Reward: {reward:.2f}, Loss: {loss:.4f}, Epsilon: {epsilon:.3f}")
    
    def save_model(self, model, name: str):
        torch.save(model.state_dict().copy(), self.weights_dir / f'{name}.pt')
                
    def plot_results(self, window: int = 100):
        
        # Plot rewards
        plt.figure(figsize=(10, 6))
        avg_rewards = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
        plt.plot(avg_rewards)
        plt.title('Average total reward across episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.log_dir / 'average_rewards.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        # avg_losses = np.convolve(self.losses, np.ones(window)/window, mode='valid')
        plt.plot(self.losses)
        plt.title('Training Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.savefig(self.log_dir / 'training_losses.png')
        plt.close()
        
        # Save raw data
        np.save(self.log_dir / 'rewards.npy', self.rewards)
        np.save(self.log_dir / 'losses.npy', self.losses)