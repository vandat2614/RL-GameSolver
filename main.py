import argparse
from src import EnvRegistry, train, test, demo
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training, Testing, and Demo')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='train, test or demo the agent')
    parser.add_argument('--config_path', type=str, default='configs/flappy_bird_v0.yaml',
                        help='path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to saved model (required for test and demo mode)')
    parser.add_argument('--save', action='store_true',
                        help='save video during demo (only applicable in demo mode)')
    return parser.parse_args()


def setup(config_path : str, mode : str):
    with open(config_path ,'r') as file:
        config = yaml.safe_load(file)
    
    config['env']['params']['render_mode'] = 'human' if mode == 'demo'  else 'rgb_array' 

    env = EnvRegistry.make(config["env"]["id"], **config["env"]["params"])
    return env, config



def main():
    args = parse_args()
    env, config = setup(args.config_path, args.mode)

    if args.mode == 'train':
        train(env, config)
    elif args.mode == 'test':
        if args.model_path is None:
            raise ValueError("Model path must be provided for test mode")
        test(env, config, args.model_path, num_episodes=100)
    elif args.mode == 'demo':
        if args.model_path is None:
            raise ValueError("Model path must be provided for demo mode")
        demo(env, config, args.model_path, save=args.save)


if __name__ == "__main__":
    main()