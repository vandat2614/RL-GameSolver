import random
import torch
import torch.nn as nn

from utils import Logger, ReplayBuffer

def update(model: nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, gamma: float, device: torch.device):
    
    states, actions, rewards, next_states, dones = batch
    
    # Convert to tensors
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
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
    
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return q_values.argmax().item()

def train(env, model: nn.Module, config: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training parameters
    num_episodes = config['train']['num_episodes']
    batch_size = config['train']['batch_size']
    gamma = config['train']['gamma']  
    epsilon_start = config['train']['epsilon_start']
    epsilon_end = config['train']['epsilon_end']
    epsilon_decay = config['train']['epsilon_decay']
    learning_rate = config['train']['learning_rate']
    
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(config['buffer_size'])
    logger = Logger(config=config)

    # Training loop
    epsilon = epsilon_start
    best_reward = float('-inf')

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:

            action = select_action(state, model, env, epsilon, device)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            # Train model if enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = update(model, batch, optimizer, criterion, gamma, device)
        
        if total_reward > best_reward:
            best_reward = total_reward
            logger.save_model(model.state_dict().copy(), 'best')

        logger.log_episode(episode + 1, total_reward, loss, epsilon)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    logger.save_model(model.state_dict(), 'last')
    logger.plot_results()