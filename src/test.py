import torch
from .models import NeuralNetwork
from .utils import Logger

def test(env, config: dict, model_path: str, num_episodes: int = 5):
    logger = Logger(config, mode='test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork.from_config(config, env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        score = 0  
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            state, reward, done, info = env.step(action)
            score = int(info.get("score", score))  # Keep updating score if available
        logger.log_episode(episode + 1, score)

    logger.log_summary()
    env.close()
