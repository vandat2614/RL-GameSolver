import argparse
from utils import NeuralNetwork
import yaml
import gymnasium

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='train or test the agent')
    parser.add_argument('--config', type=str, default='configs/flappy_bird_v0.yaml',
                      help='path to config file')
    parser.add_argument('--model-path', type=str,
                      help='path to saved model (required for test mode)',
                      default=None)
    return parser.parse_args()

def setup(config_path : str):
    with open(config_path ,'r') as file:
        config = yaml.safe_load(file)

    env = gymnasium.make(config["id"], **config["params"])
    network = NeuralNetwork.from_config(config_path, env)
    return env, network

def main():
    args = parse_args()
    env, model = setup(args.config_path)

    if args.model == 'train':
        pass
    else:
        if args.model_path is None:
            raise ValueError("Model path must be provided for test mode")
        pass

if __name__ == "__main__":
    main()