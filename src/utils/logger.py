import csv
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

        self.mode = mode
        self.rewards = []
        self.scores = []
        self.losses = []
        self.epsilons = []

        if mode == 'train':
            self.weights_dir = self.log_dir / 'weights'
            self.weights_dir.mkdir(parents=True, exist_ok=True)

            self.plots_dir = self.log_dir / 'plots'
            self.plots_dir.mkdir(parents=True, exist_ok=True)

            self.data_dir = self.log_dir / 'data'
            self.data_dir.mkdir(parents=True, exist_ok=True)

            with open(self.log_dir / 'config.json', 'w') as file:
                json.dump(config, file, indent=4)

        elif mode == 'test':
            self.test_csv_path = self.log_dir / "test_scores.csv"
            with open(self.test_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Score"])
            self.summary_path = self.log_dir / "summary.json"

        elif mode == 'demo':
            self.videos_dir = self.log_dir / 'videos'
            self.videos_dir.mkdir(parents=True, exist_ok=True)


    def log_episode(self, episode: int, score : float, total_reward: float = None, loss: float = None, epsilon: float = None):

        self.scores.append(score)
        self.rewards.append(total_reward)

        if self.mode == 'train':
            self.losses.append(loss)
            self.epsilons.append(epsilon)

            if len(self.scores) % 20 == 0:

                avg_reward = np.mean(self.rewards[-100:])
                avg_score = np.mean(self.scores[-100:])

                print(f"Episode {episode}, Avg reward: {avg_reward:.2f}, Avg score: {avg_score}, Loss: {loss:.4f}, Epsilon: {epsilon:.3f}")

        if self.mode == 'test':
            with open(self.test_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, score])
                print(f"Episode {episode}, Score: {score}")


    def log_summary(self):
        if self.mode == 'train':
            raise ValueError("Summary logging is only available in test mode")
        
        summary = {
            "min": float(np.min(self.scores)),
            "max": float(np.max(self.scores)),
            "avg": float(np.mean(self.scores)),
            "std": float(np.std(self.scores)),
            "num_episodes": len(self.scores),
        }
        with open(self.summary_path, 'w') as file:
            json.dump(summary, file, indent=4)

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

    def get_video_dir(self):
        if self.mode not in ['test', 'demo']:
            raise ValueError("Video directory only available in test or demo mode")
        return self.videos_dir
