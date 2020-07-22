import gym

from td3.training_gym import TD3_Training_Gym
from dqn.dqn import DQNLearner

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def main():
    # Start TD3 Gym Training
    start_td3()

    # Start DQN Gym Training

    #start_dqn()

def start_td3():
    env = gym.make("Pendulum-v0")

    # Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, render=False, load=False)
    env.close()

def start_dqn():
    env = gym.make("CartPole-v1")

    params = {}
    params["nr_actions"] = env.action_space.n
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    # Hyperparameters
    params["gamma"] = 0.99
    params["alpha"] = 0.001
    params["episodes"] = 500
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

if __name__ == '__main__':
    main()
