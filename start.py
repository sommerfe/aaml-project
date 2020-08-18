import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from dqn.naive_approach import DQNLearner
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import logger
from stable_baselines.bench import Monitor
import os
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_env2(env_id, rank, seed=0):
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

    #set_global_seeds(seed)
    return _init
def main():
    # Start DQN Gym Training

    start_dqn()


def start_dqn():
    env = gym.make("MountainCar-v0")

    params = {}
    params["nr_actions"] = env.action_space.n
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    # Hyperparameters
    params["gamma"] = 0.99
    params["alpha"] = 0.001
    params["episodes"] = 50
    params["epsilon"] = 0.1
    params["memory_capacity"] = 5000
    params["warmup_phase"] = 1000
    params["target_update_interval"] = 1000
    params["minibatch_size"] = 32
    params["epsilon_linear_decay"] = 1.0 / params["memory_capacity"]
    params["epsilon_min"] = 0.0001
    params["multi_processing"] = True

    training = DQNLearner(params, experiment_name='dqn_multi')
    training.start_training(env, render=False, load=False)
    env.close()

if __name__ == '__main__':
    main()


