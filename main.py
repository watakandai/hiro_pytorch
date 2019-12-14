import argparse
from train_hiro import HiroTrainer
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.models import HiroAgent
from hiro.hiro_utils import LowReplayBuffer, HighReplayBuffer, Subgoal


def run_evaluation(args):
    env = EnvWithGoal(create_maze_env(args.env_name), args.env_name)
    subgoal = Subgoal()

    goal_dim = 2
    state_dim = env.observation_space.shape[0]
    subgoal_dim = subgoal.action_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action_low = env.action_space.high
    max_action_high = subgoal.action_space.high

    scale_low = max_action_low * np.ones(action_dim)
    scale_high = max_action_high * np.ones(subgoal_dim)

    agent = HIROAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        subgoal_dim=subgoal_dim,
        scale_low=scale_low,
        scale_high=scale_high
        model_path='_'.join([args.model_path, args.date]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        low_buffer_freq=args.low_buffer_freq,
        high_buffer_freq=args.high_buffer_freq,
        low_train_freq=args.low_train_freq,
        high_train_freq=args.high_train_freq
        )
    agent.load(args.timestep)

    rewards = agent.evaluate_policy(env, args.eval_episodes, args.render, args.save_video, args.sleep)
    mean = np.mean(rewards)
    std = np.std(rewards)
    median = np.median(rewards)

    print('mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
        mean=mean, std=std, median=median))

    env.close()

def run_training(args):
    subgoal = Subgoal()

    goal_dim = 2
    state_dim = env.observation_space.shape[0]
    subgoal_dim = subgoal.action_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action_low = env.action_space.high
    max_action_high = subgoal.action_space.high

    scale_low = max_action_low * np.ones(action_dim)
    scale_high = max_action_high * np.ones(subgoal_dim)

    print('States: %i'%(state_dim))
    print('Actions: %i'%(action_dim))

    agent = HIROAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        subgoal_dim=subgoal_dim,
        scale_low=scale_low,
        scale_high=scale_high
        model_path='_'.join([args.model_path, args.date]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        low_buffer_freq=args.low_buffer_freq,
        high_buffer_freq=args.high_buffer_freq,
        low_train_freq=args.low_train_freq,
        high_train_freq=args.high_train_freq
        )

    logpath_w_date = '_'.join([args.log_path, args.date])
    logger = Logger(log_path=logpath_w_date)

    run_train_loop(
        args,
        env,
        subgoal,
        agent,
        logger)

def run_train_loop(args, env, subgoal, agent, logger):
    accum_steps = 1

    for e in np.arange(args.episode)+1:
        obs = env.reset()
        fg = obs['desired_goal']
        s = obs['observation']
        # Take Higher Action
        sg = subgoal.action_space.sample()
        done = False
        steps = 1
        episode_reward = 0
        buf = [s, fg, ]

        while not done:
            # Take Lower Action
            if accum_steps <= args.start_training_steps:
                a = env.action_space.sample()
            else:
                a = agent.choose_action_with_noise(s, sg)

            # Take Env Step
            obs, r, done, _ = env.step(a)
            d = float(done) if steps < env._max_episode_steps else 0
            n_s = obs['observation']

            # Take Higher Action
            if accum_steps <= args.start_training_steps:
                n_sg = subgoal.action_space.sample()
            else:
                n_sg = agent.choose_subgoal_with_noise(accum_steps, s, fg, sg)

            agent.append(accum_steps, s, fg, a, sg, n_s, n_sg, r, d)

            if accum_steps > args.start_training_steps:
                losses = agent.train(accum_steps)

                if _is_update(steps, args.writer_freq):
                    for k, v in losses.items():
                        logger.write('loss/%s'%(k), v, accum_steps)

            if _is_update(accum_steps, args.model_save_freq):
                agent.save(timestep=accum_steps)
                rewards = agent.evaluate_policy(env, args.eval_episodes)
                mean = np.mean(rewards)
                std = np.std(rewards)
                median = np.median(rewards)

                print('mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
                    mean=mean, std=std, median=median))

            s = n_s
            sg = n_sg
            episode_reward += r
            steps += 1
            accum_steps += 1

        logger.write('reward', episode_reward, e)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--eval_episodes', type=float, default=5)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)

    # Training
    parser.add_argument('--episode', default=1000, type=int)
    parser.add_argument('--start_training_steps', default=1000, type=int)
    parser.add_argument('--writer_freq', default=10, type=int)
    # Training (Model Saving)
    parser.add_argument('--timestep', default=-1, type=int)
    parser.add_argument('--model_save_freq', default=1000, type=int)
    # Model
    parser.add_argument('--model_path', default='model/td3_updated', type=str)
    parser.add_argument('--log_path', default='log/td3_updated', type=str)
    parser.add_argument('--actor_lr', default=0.0001, type=int)
    parser.add_argument('--critic_lr', default=0.0001, type=int)
    parser.add_argument('--expl_noise', default=0.1, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=int)
    parser.add_argument('--noise_clip', default=0.5, type=int)
    parser.add_argument('--gamma', default=0.99, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--tau', default=0.005, type=int)
    # Replay Buffer
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--low_buffer_freq', default=1, type=int)
    parser.add_argument('--high_buffer_freq', default=10, type=int)
    parser.add_argument('--low_train_freq', default=1, type=int)
    parser.add_argument('--high_train_freq', default=10, type=int)

    args = parser.parse_args()

    if args.train:
        run_training(args)

    if args.eval:
        run_evaluation(args)
