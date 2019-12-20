import argparse
import numpy as np
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.models import HiroAgent, TD3Agent
from hiro.hiro_utils import Subgoal
from hiro.utils import Logger, _is_update

def spawn_env(args, env):
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    return state_dim, goal_dim, action_dim, scale

def spawn_hiro(args, env, subgoal):
    state_dim, goal_dim, action_dim, scale_low = spawn_env(args, env)

    subgoal_dim = subgoal.action_dim
    scale_high = subgoal.action_space.high * np.ones(subgoal_dim)

    agent = HiroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        subgoal_dim=subgoal_dim,
        scale_low=scale_low,
        scale_high=scale_high,
        model_path='_'.join([args.model_path, args.filename]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        buffer_freq=args.hbuffer_freq,
        train_freq=args.train_freq,
        reward_scaling=args.reward_scaling,
        policy_freq_high=args.policy_freq_high,
        policy_freq_low=args.policy_freq_low
        )
    return agent

def spawn_td3(args, env):
    state_dim, goal_dim, action_dim, scale = spawn_env(args, env)
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        scale=scale,
        model_path='_'.join([args.model_path, args.filename]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        )
    return agent

def print_evaluation(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    median = np.median(rewards)

    print('mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
        mean=mean, std=std, median=median))

def run_evaluation(args, spawn_agent):
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    subgoal = Subgoal()

    agent = spawn_agent(args, env)
    agent.load(args.timestep)

    rewards = agent.evaluate_policy(env, subgoal, args.eval_episodes, args.render, args.save_video, args.sleep)
    print_evaluation(rewards)

def run_hiro_training(args):
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    subgoal = Subgoal()

    agent = spawn_hiro(args, env, subgoal)

    logpath = '_'.join([args.log_path, args.filename])
    logger = Logger(log_path=logpath)

    accum_steps = 0

    for e in np.arange(args.episode):
        obs = env.reset()
        fg = obs['desired_goal']
        s = obs['observation']
        # Take Higher Action
        sg = subgoal.action_space.sample()
        agent.set_final_goal(fg)
        done = False
        steps = 0
        episode_reward = 0
        episode_subreward = 0

        while not done:
            # Take Lower Action
            if accum_steps <= args.start_training_steps:
                a = env.action_space.sample()
            else:
                a = agent.choose_action_with_noise(s, sg)

            # Take Env Step
            obs, r, done, _ = env.step(a)
            d = float(done)
            n_s = obs['observation']

            # Take Higher Action
            if accum_steps <= args.start_training_steps:
                n_sg = subgoal.action_space.sample()
            else:
                n_sg = agent.choose_subgoal_with_noise(steps, s, sg, n_s)
            
            sr = agent.low_reward(s, sg, n_s)
            agent.append(steps, s, a, sg, n_s, n_sg, r, sr, d)

            if accum_steps >= args.start_training_steps:
                losses, td_errors = agent.train(accum_steps)

                if args.save and _is_update(steps, args.writer_freq):
                    for k, v in losses.items():
                        logger.write('loss/%s'%(k), v, accum_steps)
                    
                    for k, v in td_errors.items():
                        logger.write('td_error/%s'%(k), v, accum_steps)

            if _is_update(accum_steps, args.model_save_freq):
                if args.save:
                    agent.save(timestep=accum_steps)
                rewards = agent.evaluate_policy(env, subgoal, args.eval_episodes)
                mean = np.mean(rewards)
                std = np.std(rewards)
                median = np.median(rewards)

                print('episode {episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
                    episode=e, mean=mean, std=std, median=median))

            s = n_s
            sg = n_sg
            episode_reward += r
            episode_subreward += sr
            steps += 1
            accum_steps += 1

        if args.save:
            logger.write('reward/Reward', episode_reward, e)
            logger.write('reward/Intrinsic Reward', episode_subreward, e)

def run_td3_training(args):
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    subgoal = Subgoal()

    agent = spawn_td3(args, env)

    logpath = '_'.join([args.log_path, args.filename])
    logger = Logger(log_path=logpath)

    accum_steps = 0

    for e in np.arange(args.episode):
        obs = env.reset()
        fg = obs['desired_goal']
        s = obs['observation']
        done = False
        steps = 0
        episode_reward = 0

        while not done:
            # Take Lower Action
            if accum_steps <= args.start_training_steps:
                a = env.action_space.sample()
            else:
                a = agent.choose_action_with_noise(s, fg)

            # Take Env Step
            obs, r, done, _ = env.step(a)
            d = float(done)
            n_s = obs['observation']

            agent.append(steps, s, fg, a, n_s, r, d)

            if accum_steps >= args.start_training_steps:
                losses, td_errors = agent.train(accum_steps)

                if args.save and _is_update(steps, args.writer_freq):
                    for k, v in losses.items():
                        logger.write('loss/%s'%(k), v, accum_steps)
                    
                    for k, v in td_errors.items():
                        logger.write('td_error/%s'%(k), v, accum_steps)

            if _is_update(accum_steps, args.model_save_freq):
                if args.save:
                    agent.save(timestep=accum_steps)
                rewards = agent.evaluate_policy(env, subgoal, args.eval_episodes)
                mean = np.mean(rewards)
                std = np.std(rewards)
                median = np.median(rewards)

                print('episode {episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
                    episode=e, mean=mean, std=std, median=median))

            s = n_s
            episode_reward += r
            steps += 1
            accum_steps += 1

        if args.save:
            logger.write('reward/Reward', episode_reward, e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--eval_episodes', type=float, default=5)
    parser.add_argument('--env', default='AntMaze', type=str)
    parser.add_argument('--td3', action='store_true')

    # Training
    parser.add_argument('--episode', default=25000, type=int)
    parser.add_argument('--start_training_steps', default=1000, type=int)
    parser.add_argument('--writer_freq', default=25, type=int)
    # Training (Model Saving)
    parser.add_argument('--save', action='store_false')
    parser.add_argument('--timestep', default=-1, type=int)
    parser.add_argument('--model_save_freq', default=10000, type=int)
    # Model
    parser.add_argument('--model_path', default='model/hiro', type=str)
    parser.add_argument('--log_path', default='log/hiro', type=str)
    parser.add_argument('--actor_lr', default=0.0001, type=int)
    parser.add_argument('--critic_lr', default=0.0001, type=int)
    parser.add_argument('--expl_noise', default=0.1, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=int)
    parser.add_argument('--noise_clip', default=0.5, type=int)
    parser.add_argument('--gamma', default=0.99, type=int)
    parser.add_argument('--tau', default=0.005, type=int)
    parser.add_argument('--policy_freq_low', default=2, type=int)
    parser.add_argument('--policy_freq_high', default=2, type=int)
    # Replay Buffer
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--buffer_freq', default=10, type=int)
    parser.add_argument('--train_freq', default=10, type=int)
    parser.add_argument('--reward_scaling', default=0.1, type=float)

    args = parser.parse_args()

    if args.train:
        if args.td3:
            run_td3_training(args)
        else:
            run_hiro_training(args)

    if args.eval:
        if args.td3:
            run_evaluation(args, spawn_td3)
        else:
            run_evaluation(args, spawn_hiro)
