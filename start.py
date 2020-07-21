import gym

from td3.training_gym import TD3_Training_Gym

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def main():
    # Start TD3 Gym Training
    start_gym_std()


def start_gym_std():
    # env = gym.make("CartPole-v1")
    env = gym.make("Pendulum-v0")

    # Gym version with render
    training = TD3_Training_Gym()
    training.start_training(env, render=False, load=False)
    env.close()


if __name__ == '__main__':
    main()
