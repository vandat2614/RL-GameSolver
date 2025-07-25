import random
import torch
import torch.nn as nn
import math

from .utils import Logger, ReplayBuffer
from .models import NeuralNetwork, ConvNeuralNetwork
from gymnasium.spaces import Tuple as SpaceTuple

def update(model: nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, gamma: float, device: torch.device):
    
    batch = list(zip(*batch))
    states, actions, rewards, next_states, dones = batch

    if isinstance(model, ConvNeuralNetwork):
        num_branches = len(states[0])  
        states = [
            torch.stack([
                torch.tensor(state[i]).permute(2, 0, 1) for state in states
            ]) for i in range(num_branches)
        ]
        next_states = [
            torch.stack([
                torch.tensor(next_state[i]).permute(2, 0, 1) for next_state in next_states
            ]) for i in range(num_branches)
        ]
        
        states = [s.to(device).float() for s in states]
        next_states = [s.to(device).float() for s in next_states]

    else:
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute Q values
    current_q_values = model(states).gather(dim=1, index=actions.unsqueeze(1))
    next_q_values = model(next_states).max(dim=1)[0].detach()
    target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    # Update model
    loss = criterion(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def select_action(state: torch.Tensor, model: nn.Module, env, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return env.action_space.sample()
    
    # print("CALL THIS")

    with torch.no_grad():

        if isinstance(state, tuple):
            num_branches = len(state)
            state_tensor = [torch.tensor(state[i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                            for i in range(num_branches)]
            if num_branches == 1:
                state_tensor = state_tensor[0]
        else: 
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return q_values.argmax().item()

def train(env, config: str):

    if config["model"]["type"] == "mlp":
        model = NeuralNetwork.from_config(config, env.observation_space.shape[0], env.action_space.n)
    else: 
    
        obs_space = env.observation_space
        if isinstance(obs_space, SpaceTuple):
            input_sizes = [obs.shape for obs in obs_space]
        else: input_sizes = [obs_space.shape]

        output_size = env.action_space.n
        model = ConvNeuralNetwork.from_config(config, input_sizes, output_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training parameters
    num_episodes = config['train']['num_episodes']
    batch_size = config['train']['batch_size']
    gamma = config['train']['gamma']  
    epsilon_start = config['train']['epsilon_start']
    epsilon_end = config['train']['epsilon_end']
    epsilon_decay = config['train']['epsilon_decay']
    warmup_episodes = config['train']['warmup_episodes']
    learning_rate = config['train']['learning_rate']
    
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(config['train']['buffer_size'])
    logger = Logger(config=config, mode="train")

    # Training loop
    epsilon = epsilon_start
    best_score = float('-inf')

    # import time

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = truncated = False
        loss = None

        while not (done or truncated):

            action = select_action(state, model, env, epsilon, device)
            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state


            # Train model if enough samples
            # if len(replay_buffer) >= batch_size:
            #     batch = replay_buffer.sample(batch_size)
            #     loss = update(model, batch, optimizer, criterion, gamma, device)
        
            # print(f"Step: {env.num_steps}, Reward: {reward:.2f}", end="")
            # if loss is not None:
                # print(f", Loss: {loss:.4f}")
            # print()

            env.render()
            # time.sleep(1)

        score =  info["score"]
        print(f"End episode {episode}, score : {score}")

        if score > best_score:
            best_score = score
            logger.save_model(model, 'best')

        logger.log_episode(episode + 1, score, total_reward, loss, epsilon)

        if episode + 1 < warmup_episodes:
            epsilon = epsilon_start
        else:
            epsilon = max(epsilon_end, epsilon_start * math.exp(-epsilon_decay * (episode - warmup_episodes)))

    logger.save_model(model, 'last')
    logger.plot_results()