import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

class Logger:
    def __init__(self, config: dict, mode: str = 'train'):
        log_dir = config.get("logger", {}).get("log_dir", "runs")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / mode / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if mode == 'train':
            self.weights_dir = self.log_dir / 'weights'
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            
            self.plots_dir = self.log_dir / 'plots'
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir = self.log_dir / 'data'
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_dir / 'config.yaml', 'w') as file:
                json.dump(config, file, indent=4)
        else:
            self.videos_dir = self.log_dir / 'videos'
            self.videos_dir.mkdir(parents=True, exist_ok=True)
        
        self.rewards = []
        self.losses = []
        self.epsilons = []
        self.mode = mode
    
    def get_video_dir(self):
        if self.mode != 'test':
            raise ValueError("Video directory only available in test mode")
        return self.videos_dir

    def log_episode(self, episode: int, reward: float, loss: float = None, epsilon: float = None, show = False):
        self.rewards.append(reward)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        if loss is not None:
            self.losses.append(loss)

        if not show:
            return

        if self.mode == 'train':
            avg_reward = np.mean(self.rewards[-100:])
            if loss is not None:
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Loss: {loss:.4f}, Epsilon: {epsilon:.3f}")
            else:
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        else:
            print(f"Test Episode {episode}, Reward: {reward:.2f}")

    def save_model(self, model, name: str):
        if self.mode != 'train':
            raise ValueError("Model saving only available in training mode")
        torch.save(model.state_dict(), self.weights_dir / f'{name}.pt')

    def plot_results(self, window: int = 100):
        if self.mode != 'train':
            raise ValueError("Plotting only available in training mode")
            
        plt.figure(figsize=(10, 6))
        avg_rewards = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
        plt.plot(avg_rewards)
        plt.title('Average total reward across episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.plots_dir / 'rewards.png')
        plt.close()
        
        if self.losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses)
            plt.title('Training Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.savefig(self.plots_dir / 'losses.png')
            plt.close()
        
        np.save(self.data_dir / 'rewards.npy', self.rewards)
        if self.losses:
            np.save(self.data_dir / 'losses.npy', self.losses)
        if self.epsilons:
            np.save(self.data_dir / 'epsilons.npy', self.epsilons)