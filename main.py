import argparse
from src import EnvRegistry, train, test
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='train or test the agent')
    parser.add_argument('--config_path', type=str, default='configs/flappy_bird_v0.yaml',
                      help='path to config file')
    parser.add_argument('--model_path', type=str,
                      help='path to saved model (required for test mode)',
                      default=None)
    return parser.parse_args()

def setup(config_path : str, mode : str):
    with open(config_path ,'r') as file:
        config = yaml.safe_load(file)
    
    config['env']['params']['render_mode'] = None if mode == 'train' else 'rgb_array'

    env = EnvRegistry.make(config["env"]["id"], **config["env"]["params"])
    return env, config

def main():
    args = parse_args()
    env, config = setup(args.config_path, args.mode)

    if args.mode == 'train':
        train(env, config)
    else:
        if args.model_path is None:
            raise ValueError("Model path must be provided for test mode")
        test(env, config, args.model_path)

if __name__ == "__main__":
    main()