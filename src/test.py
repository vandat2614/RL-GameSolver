import torch
from .models import NeuralNetwork
from .utils import Logger
from gymnasium.wrappers import RecordVideo

def test(env, config : str, model_path: str, num_episodes: int = 5):

    logger = Logger(config, mode='test')
    env = RecordVideo(
        env,
        video_folder=str(logger.get_video_dir()),
        episode_trigger=lambda ep_id: True,  # Record all episodes
        name_prefix="rl-video"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork.from_config(config, env)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Test loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Take action
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        logger.log_episode(episode + 1, total_reward)
    
    env.close()