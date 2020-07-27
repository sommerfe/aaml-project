import time

import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2, A2C, ACER
import neptune
from env_variable import neptune_api_token

environments = ["MountainCar-v0", "MountainCarContinuous-v0", "FrozenLake8x8-v0", "HotterColder-v0"]
algorithms = ["ppo2", "a2c", "acktr"]
multiprocessing = [True, False]
cores = [2, 4, 6, 8]
timesteps = 50000

iterations = 100


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(model.env.num_envs)]
    obs = model.env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = model.env.step(actions)

        # Stats
        for i in range(model.env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

    mean_rewards = [0.0 for _ in range(model.env.num_envs)]
    n_episodes = 0
    for i in range(model.env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])
        n_episodes += len(episode_rewards[i])

        # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward

def main():

    for multi in multiprocessing:
        for envi in environments:
            for algo in algorithms:
                if multi:
                    for n_core in cores:
                        experiment_name = 'multi_' + envi + '_' + algo + '_' + str(n_core)
                        params = {"multi": multi,
                                  "environment": envi,
                                  "algorithm": algo,
                                  "cores": n_core,
                                  "timesteps": timesteps}
                        neptune.init('sommerfe/aaml-project', neptune_api_token)
                        neptune.create_experiment(experiment_name, params=params)
                        duration = []
                        for i in range(iterations):
                            if multi:
                                env = SubprocVecEnv([make_env(envi, i) for i in range(n_core)])
                            else:
                                env = DummyVecEnv([lambda: gym.make(envi)])

                            if algo == "ppo2":
                                model = PPO2(MlpPolicy, env, verbose=0)
                            elif algo == "acktr":
                                model = ACKTR(MlpPolicy, env, verbose=0)
                            else:
                                model = A2C(MlpPolicy, env, verbose=0)

                            duration.append(training(model, timesteps))
                        neptune.log_text('duration_avg', str(sum(duration) / len(duration)))
                else:
                    experiment_name = 'single_' + envi + '_' + algo
                    params = {"multi": multi,
                              "environment": envi,
                              "algorithm": algo,
                              "timesteps": timesteps}
                    neptune.init('sommerfe/aaml-project', neptune_api_token)
                    neptune.create_experiment(experiment_name, params=params)
                    duration = []
                    for i in range(iterations):
                        if algo == "ppo2":
                            model = PPO2(MlpPolicy, env, verbose=0)
                        elif algo == "acktr":
                            model = ACKTR(MlpPolicy, env, verbose=0)
                        else:
                            model = A2C(MlpPolicy, env, verbose=0)
                        training_single_core(model, timesteps)
                        duration.append(training(model, timesteps))
                    neptune.log_text('duration_avg', str(sum(duration) / len(duration)))


def training(model, n_timesteps=25000):
    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(
        "Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))
    neptune.log_metric('training_duration', total_time_multi)
    neptune.log_metric('training_fps', n_timesteps / total_time_multi)
    neptune.log_metric('reward', evaluate(model))

    return total_time_multi

def training_single_core(model, n_timesteps=25000):
    # Single Process RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single,
                                                                        n_timesteps / total_time_single))
    neptune.log_metric('training_duration', total_time_single)
    neptune.log_metric('training_fps', n_timesteps / total_time_single)
    neptune.log_metric('reward', evaluate(model))

    return total_time_single

if __name__ == '__main__':
    main()
