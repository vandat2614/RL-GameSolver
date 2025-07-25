import os
import torch
import cv2
import numpy as np
import pygame
import pygame.surfarray
from .models import NeuralNetwork, ConvNeuralNetwork
from .utils import Logger
from gymnasium.spaces import Tuple as SpaceTuple

def demo(env, config: dict, model_path: str, save=False):
    logger = Logger(config, mode="demo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model"]["type"] == "mlp":
        model = NeuralNetwork.from_config(config, env.observation_space.shape[0], env.action_space.n)
    else: 
        obs_space = env.observation_space
        if isinstance(obs_space, SpaceTuple):
            input_sizes = [obs.shape for obs in obs_space]
        else: input_sizes = [obs_space.shape]

        output_size = env.action_space.n
        model = ConvNeuralNetwork.from_config(config, input_sizes, output_size)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    if save:
        video_path = os.path.join(logger.get_video_dir(), "episode.mp4")
        surface = env.unwrapped._surface
        width, height = surface.get_size()
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():


            if config["model"]["type"] == "mlp":
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            elif config["model"]["type"] == "cnn":
                if isinstance(obs_space, SpaceTuple):
                    num_branches = len(state)
                    state_tensor = [torch.tensor(state[i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                                    for i in range(num_branches)]
                    if num_branches == 1:
                        state_tensor = state_tensor[0]

            q_values = model(state_tensor)
            action = q_values.argmax().item()

        state, reward, done, info = env.step(action)

        env.render()

        if save:
            surface = env.unwrapped._surface
            frame_rgb = np.array(pygame.surfarray.array3d(surface)).swapaxes(0, 1)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    print(info)

    if save:
        writer.release()

    env.close()
