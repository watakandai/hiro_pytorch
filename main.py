import argparse
from train_hiro import HiroTrainer 
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.models import HiroAgent

def run_training(args):
    # +alpha Some Configurations
    env = EnvWithGoal(create_maze_env(args.env_name), args.env_name)
    trainer = HiroTrainer(env, args)
    trainer.train(args.render, args.sleep)

def play(args):
    # +alpha Some Configurations
    env = EnvWithGoal(create_maze_env(args.env_name), args.env_name)
    agent = HiroAgent(env)
    agent.load()
    agent.play(render=args.render, sleep=args.sleep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=0, type=int)                      # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env_name", default="AntMaze", type=str)    # Environment name
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    args = parser.parse_args()

    if args.train:
        run_training(args)
    elif args.play:
        play(args)
    else:
        raise ValueError('No Argument Passed')