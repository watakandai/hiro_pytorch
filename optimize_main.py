import argparse
import numpy as np
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.hiro_utils import Subgoal
from hiro.utils import Logger, _is_update
from main import spawn_env
from hiro.models import HiroAgent
import datetime

import optuna

class HyperParamters():
    def __init__(self, trial):
        self.policy_freq_low = trial.suggest_categorical('policy freq low', [1,2])
        self.policy_freq_high = trial.suggest_categorical('policy freq high', [1,2])
        self.buffer_size = int(trial.suggest_uniform('buffer size', 10000, 1000000))
        self.batch_size = int(trial.suggest_loguniform('batch size', 32, 1000))
        self.reward_scaling = trial.suggest_categorical('Reward Scaling', [0.1, 1])
        self.buffer_freq = int(trial.suggest_uniform('buffer freq', 2, 100))

class Objective():
    def __init__(self, args):
        self.args = args

    def __call__(self, trial):
        args = self.args
        hyps = HyperParamters(trial)

        env = EnvWithGoal(create_maze_env(args.env), args.env)
        subgoal = Subgoal()

        state_dim, goal_dim, action_dim, scale_low = spawn_env(args, env)

        subgoal_dim = subgoal.action_dim
        scale_high = subgoal.action_space.high * np.ones(subgoal_dim)

        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d_%H%M%S')

        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            scale_low=scale_low,
            scale_high=scale_high,
            model_path='_'.join([args.model_path, filename]),
            buffer_size=hyps.buffer_size,
            batch_size=hyps.batch_size,
            buffer_freq=hyps.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=hyps.reward_scaling,
            policy_freq_high=hyps.policy_freq_high,
            policy_freq_low=hyps.policy_freq_low
            )

        logpath = '_'.join([args.log_path, filename])
        logger = Logger(log_path=logpath)

        accum_steps = 0
        best_reward = -np.inf

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
                    rewards = agent.evaluate_policy(env, subgoal)
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

            if accum_steps >= args.start_training_steps:
                if episode_reward > best_reward:
                    best_reward = episode_reward

        return best_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pruning', action='store_true')
    ###### No Change
    # Training
    parser.add_argument('--env', default='AntMaze', type=str)
    parser.add_argument('--model_path', default='model/hiro', type=str)
    parser.add_argument('--log_path', default='log/hiro', type=str)
    parser.add_argument('--episode', default=25000, type=int)
    parser.add_argument('--start_training_steps', default=1000, type=int)
    parser.add_argument('--writer_freq', default=25, type=int)
    # Training (Model Saving)
    parser.add_argument('--save', action='store_false')
    parser.add_argument('--timestep', default=-1, type=int)
    parser.add_argument('--model_save_freq', default=10000, type=int)
    # Replay Buffer
    parser.add_argument('--train_freq', default=10, type=int)

    args = parser.parse_args()

    objective = Objective(args)
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NoPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))