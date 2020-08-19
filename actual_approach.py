import time
import gym
import numpy as np
import psutil
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2, A2C
#import neptune
#from env_variable import neptune_api_token, neptune_project_name
import random

"""
Parts of the code are from https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82
This page is also referenced in the report.
Note that all neptune logging parts have been commented out, so there is no logging currently active.
This script was executed on Python 3.6.
"""

environments = ["MountainCar-v0", "MountainCarContinuous-v0", "FrozenLake8x8-v0", "HotterColder-v0"]
environments2 = ["CartPole-v0", "CartPole-v1"]
algorithms = ["ppo2", "a2c", "acktr"]
multiprocessing = [True, False]
parallel_environments = [2]
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

    return mean_reward

def all_environments():
    """
    Calculates the results of the all environments section in the report.
    For every variation of multiprocessing, environment, algorithm and number of cores one experiment is started.
    """

    #neptune.init(neptune_project_name, neptune_api_token)
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
                        #neptune.create_experiment(experiment_name, params=params)
                        #neptune.log_text('cpu_count', str(psutil.cpu_count()))
                        #neptune.log_text('count_non_logical', str(psutil.cpu_count(logical=False)))
                        duration = []
                        for i in range(iterations):
                            env = SubprocVecEnv([make_env(envi, i) for i in range(n_core)])
                            if algo == "ppo2":
                                model = PPO2(MlpPolicy, env, verbose=0)
                            elif algo == "acktr":
                                model = ACKTR(MlpPolicy, env, verbose=0)
                            else:
                                model = A2C(MlpPolicy, env, verbose=0)

                            duration.append(training(model, timesteps))
                        #neptune.log_text('duration_avg', str(sum(duration) / len(duration)))
                else:
                    env = DummyVecEnv([lambda: gym.make(envi)])
                    experiment_name = 'single_' + envi + '_' + algo
                    params = {"multi": multi,
                              "environment": envi,
                              "algorithm": algo,
                              "timesteps": timesteps}
                    #neptune.create_experiment(experiment_name, params=params)
                    #neptune.log_text('cpu_count', str(psutil.cpu_count()))
                    #neptune.log_text('count_non_logical', str(psutil.cpu_count(logical=False)))
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
                    #neptune.log_text('duration_avg', str(sum(duration) / len(duration)))


def training(model, n_timesteps=25000):
    """
    Trains the multi processing model.

    :param model: (BaseRLModel object) the RL Agent
    :param n_timesteps: (int) number of timesteps to train the model
    :return: (float) Total duration of the training
    """

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    #print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))
    #neptune.log_metric('training_duration', total_time_multi)
    #neptune.log_metric('training_fps', n_timesteps / total_time_multi)
    #neptune.log_metric('reward', evaluate(model))

    return total_time_multi

def training_single_core(model, n_timesteps=25000):
    """
    Trains the single processing model.

    :param model: (BaseRLModel object) the RL Agent
    :param n_timesteps: (int) number of timesteps to train the model
    :return: (float) Total duration of the training
    """

    # Single Process RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    #print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single, n_timesteps / total_time_single))
    #neptune.log_metric('training_duration', total_time_single)
    #neptune.log_metric('training_fps', n_timesteps / total_time_single)
    #neptune.log_metric('reward', evaluate(model))

    return total_time_single



def all_algorithms_parallel():
    """
    Calculates the results of the all algorithms section in the report. This method only covers the parallel parts.
    For every variation of algorithm and number of cores one experiment is started.
    """

    #neptune.init(neptune_project_name, neptune_api_token)
    for pe in parallel_environments:
        for algo in algorithms:
            experiment_name = 'pe_' + str(pe) + '_' + algo
            params = {"parallel_environments": pe,
                      "algorithm": algo,
                      "timesteps": timesteps}
            #neptune.create_experiment(experiment_name, params=params)
            #neptune.log_text('cpu_count', str(psutil.cpu_count()))
            #neptune.log_text('count_non_logical', str(psutil.cpu_count(logical=False)))
            duration = []
            for i in range(iterations):
                environment_list = random.sample(environments2, pe)
                env = SubprocVecEnv([make_env(envi, i) for i, envi in enumerate(environment_list)])
                if algo == "ppo2":
                    model = PPO2(MlpPolicy, env, verbose=0)
                elif algo == "acktr":
                    model = ACKTR(MlpPolicy, env, verbose=0)
                else:
                    model = A2C(MlpPolicy, env, verbose=0)
                duration.append(training(model, timesteps))
            #neptune.log_text('duration_avg', str(sum(duration) / len(duration)))


def all_algorithms_single():
    """
    Calculates the results of the all environments section in the report.
    For every variation of environment and algorithm one experiment is started.
    """

    #neptune.init(neptune_project_name, neptune_api_token)
    for envi in environments2:
        for algo in algorithms:
            experiment_name = 'pe_single_' + envi + '_' + algo
            params = {"environment": envi,
                      "algorithm": algo,
                      "timesteps": timesteps}
            #neptune.create_experiment(experiment_name, params=params)
            #neptune.log_text('cpu_count', str(psutil.cpu_count()))
            #neptune.log_text('count_non_logical', str(psutil.cpu_count(logical=False)))
            duration = []
            for i in range(iterations):
                env = DummyVecEnv([lambda: gym.make(envi)])
                if algo == "ppo2":
                    model = PPO2(MlpPolicy, env, verbose=0)
                elif algo == "acktr":
                    model = ACKTR(MlpPolicy, env, verbose=0)
                else:
                    model = A2C(MlpPolicy, env, verbose=0)
                training_single_core(model, timesteps)
                duration.append(training(model, timesteps))
            #neptune.log_text('duration_avg', str(sum(duration) / len(duration)))



if __name__ == '__main__':
    all_environments()
    all_algorithms_parallel()
    all_algorithms_single()
