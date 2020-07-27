import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from td3.training_unity import TD3_Training_Unity
from td3.training_gym import TD3_Training_Gym
from dqn.dqn import DQNLearner
from td3.training_unity_multi import TD3_Training_Unity_Multi
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


def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True): # pylint: disable=C0111
        def _thunk():
            no_graphics = not use_visual
            unity_env = UnityEnvironment(env_directory, no_graphics=True)
            env = UnityToGymWrapper(unity_env, uint8_visual=False)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if not visual:
        print('sub')
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        print('dummy')
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])


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
    # Start TD3 Gym Training
    #start_td3()

    # Start DQN Gym Training

    #start_dqn()

    #start_unity_td3()

    start_unity_td3_multi()

def start_td3():
    #env = gym.make("Pendulum-v0")
    env = gym.make("MountainCarContinuous-v0")

    # Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, render=False, load=False)
    env.close()

def start_unity_td3():

    unity_env = UnityEnvironment('./worm_dynamic_one_agent/win/UnityEnvironment', no_graphics=False)
    env = UnityToGymWrapper(unity_env, uint8_visual=False)
    #env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    training = TD3_Training_Unity()
    training.start_training(env, load=False)
    env.close()

def start_unity_td3_multi():
    num_env = 1
    env = SubprocVecEnv([make_env2("MountainCarContinuous-v0", i) for i in range(num_env)])
    #model = ACKTR(MlpPolicy, env, verbose=0)
    #mean_reward_before_train = evaluate(model, num_steps=1000)
    #print(mean_reward_before_train)
    #env = SubprocVecEnv([gym.make("MountainCarContinuous-v0") for i in range(num_env)])
    #env = gym.make("MountainCarContinuous-v0")
    training = TD3_Training_Gym()
    training.start_training(env, load=False)
    env.close()

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

    training = DQNLearner(params)
    training.start_training(env, render=False, load=False)
    env.close()

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

if __name__ == '__main__':
    main()


