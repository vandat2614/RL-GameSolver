import os
import torch
import cv2
import numpy as np
import pygame
import pygame.surfarray
from .models import NeuralNetwork
from .utils import Logger

def demo(env, config: dict, model_path: str, save=False):
    logger = Logger(config, mode="demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork.from_config(config, env.observation_space.shape[0], env.action_space.n)
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
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        state, reward, done, info = env.step(action)

        if save:
            surface = env.unwrapped._surface
            frame_rgb = np.array(pygame.surfarray.array3d(surface)).swapaxes(0, 1)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    if save:
        writer.release()

    env.close()
