import numpy as np
import torch
import argparse
import os
from datetime import datetime
import time
from td3.TD3 import TD3
from td3.utils import ReplayBuffer
from multi_env import Process

"""
from dev.td3 import OurDDPG
from dev.td3 import DDPG
from dev.td3.TD3 import TD3
from dev.td3.utils import ReplayBuffer, DynamicExperienceReplay

"""


class TD3_Training_Unity_Multi:
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, policy, env, seed, eval_episodes=10):
        #eval_env = gym.make(env_name)
        eval_env = env
        #eval_env.seed(seed + 100)
        eval_env.action_space.seed(seed + 100)

        avg_reward = 0.
        episode = 0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                action = np.array(action).reshape((1, 9))
                #print(eval_env.action_space)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
            print("Eval Episode:  " + str(episode))
            episode += 1

        avg_reward /= eval_episodes

        print("---------------------------------------")
        #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print(f"{datetime.now()} \t Evaluation over {eval_episodes} episodes: {avg_reward}")
        print("---------------------------------------")
        return avg_reward

    def start_training(self, env, load=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env",
                            default="Worm")  # OpenAI gym environment name (not used to start env in AlphaWorm)
        parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=1e6, type=int)  # Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e9, type=int)  # Max time steps to run environment

        parser.add_argument("--max_episodes", default=2, type=int)  # Max time steps to run environment
        parser.add_argument("--max_env_episode_steps", default=1e3, type=int)  # Max env steps

        parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
        parser.add_argument("--random_policy_ratio",
                            default=1)  # ratio of random episodes 1 = as many random as policy, 2 = double as many policy as random ...
        parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)  # Discount factor
        parser.add_argument("--tau", default=0.005)  # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
        parser.add_argument("--save_model", default=True, action="store_true")  # Save model and optimizer parameters
        if load:
            parser.add_argument("--load_model",
                                default="default")  # Model load file name, "" doesn't load, "default" uses file_name
        else:
            parser.add_argument("--load_model",
                                default="")  # Model load file name, "" doesn't load, "default" uses file_name

        parser.add_argument("--random_policy", default=False)  # Activate random policy

        self.args = parser.parse_args()

        file_name = f"{self.args.policy}_{self.args.env}_{self.args.seed}"
        print("---------------------------------------")
        print(f"{datetime.now()} \t Policy: {self.args.policy}, Env: {self.args.env}, Seed: {self.args.seed}")
        print("---------------------------------------")

        if not os.path.exists("./td3/results"):
            os.makedirs("./td3/results")

        if self.args.save_model and not os.path.exists("./td3/models"):
            os.makedirs("./td3/models")

        # Set seeds
        # env.seed(args.seed)
        env.action_space.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.args.discount,
            "tau": self.args.tau,
        }

        # Initialize policy
        if self.args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = self.args.policy_noise * max_action
            kwargs["noise_clip"] = self.args.noise_clip * max_action
            kwargs["policy_freq"] = self.args.policy_freq
            self.policy = TD3(**kwargs)

        if self.args.load_model != "":
            policy_file = file_name if self.args.load_model == "default" else self.args.load_model
            self.policy.load(f"./td3/models/{policy_file}")

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)


        # Evaluate untrained policy
        #evaluations = [self.eval_policy(policy, env, args.seed)]

        #state, done = env.reset(), False
        l = []
        # [p.join() for p in l]
        self.env = env
        tic_training = time.perf_counter()

        for e in range(int(self.args.max_episodes)):
            tic_episode = time.perf_counter()
            #p = Process(target=self.episode, args=([], file_name, max_action, action_dim, state_dim))
            #p.start()
            #l.append(p)
            self.episode([], file_name, max_action, action_dim, state_dim)
            toc_episode = time.perf_counter()
            print('episode_duration', e,':', toc_episode - tic_episode)
        toc_training = time.perf_counter()
        print('training_duration',':', toc_training - tic_training)

    def episode(self, evaluations, file_name, max_action, action_dim, state_dim):
        replay_buffer = ReplayBuffer(state_dim, action_dim)

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.args.discount,
            "tau": self.args.tau,
        }
        kwargs["policy_noise"] = self.args.policy_noise * max_action
        kwargs["noise_clip"] = self.args.noise_clip * max_action
        kwargs["policy_freq"] = self.args.policy_freq
        policy = TD3(**kwargs)

        state, done = self.env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(self.args.max_env_episode_steps)):
            episode_timesteps += 1

            if t < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * self.args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            #action = np.array(action).reshape((1, 9))
            next_state, reward, done, _ = self.env.step(action)
            done = True if episode_timesteps % self.args.max_env_episode_steps == 0 else False
            done_bool = float(done) if episode_timesteps < self.args.max_env_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            #if t >= args.start_timesteps:
            #    self.policy.train(self.replay_buffer, args.batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"{datetime.now()} \t Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward}")
                # Reset environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            #if (t + 1) % args.eval_freq == 0:
            #    evaluations.append(self.eval_policy(self.policy, self.env, args.seed))
            #    np.save(f"./td3/results/{file_name}", evaluations)
            #    if args.save_model: self.policy.save(f"./td3/models/{file_name}")
