import random
import numpy as np
import os
import neptune
from env_variable import neptune_api_token, neptune_project_name
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import time
import psutil
from multiprocessing import Process
import gym

'''
Parts of the code has been used from the Praktikum Autonome Systeme at the Ludwig-Maximilians Universität.
'''

class QLearner():

    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.action_counts = np.zeros(self.nr_actions)

    def Q(self, state):
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values

    def policy(self, state):
        return epsilon_greedy(self.Q_values, self.action_counts, epsilon=self.epsilon)

    def update(self, state, action, reward, next_state):
        pass

def epsilon_greedy(Q_values, action_counts, epsilon=0.1):
    if np.random.rand() <= epsilon:
        return random.choice(range(len(Q_values)))
    else:
        return np.argmax(Q_values)


def build_model(input_dimension, output_dimension, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=input_dimension))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_dimension, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model



class ReplayMemory:

    def __init__(self, size):
        self.transitions = []
        self.size = size

    def save(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.size:
            self.transitions.pop(0)

    def sample_batch(self, minibatch_size):
        nr_episodes = len(self.transitions)
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()

    def size(self):
        return len(self.transitions)




class DQNLearner(QLearner):
    """
     Autonomous agent using Deep Q-Learning.
    """

    def __init__(self, params, experiment_name="dqn"):
        super(DQNLearner, self).__init__(params)
        self.nr_input_features = params["nr_input_features"]
        self.epsilon = 1
        self.epsilon_linear_decay = params["epsilon_linear_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.warmup_phase = params["warmup_phase"]
        self.minibatch_size = params["minibatch_size"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.training_count = 0
        self.target_update_interval = params["target_update_interval"]
        self.policy_net = build_model(self.nr_input_features, self.nr_actions, self.alpha)
        self.target_net = build_model(self.nr_input_features, self.nr_actions, self.alpha)
        self.update_target_network()
        self.training_episodes = params["episodes"]
        self.multi_processing = params["multi_processing"]
        neptune.init(neptune_project_name, neptune_api_token)
        neptune.create_experiment(experiment_name, params=params)



    def start_training(self, env, render=False, load=False):
        if not os.path.exists("./dqn/results"):
            os.makedirs("./dqn/results")

        if not os.path.exists("./dqn/models"):
            os.makedirs("./dqn/models")

        file_name = "dqn_result"
        evaluations = []
        neptune.log_text('cpu_count', str(psutil.cpu_count()))
        neptune.log_text('count_non_logical', str(psutil.cpu_count(logical=False)))

        l = []

        tic_training = time.perf_counter()
        for i in range(self.training_episodes):
            neptune.log_text('avg_cpu_load', str(psutil.getloadavg()))
            neptune.log_text('cpu_percent', str(psutil.cpu_percent(interval=1, percpu=True)))
            tic_episode = time.perf_counter()
            if self.multi_processing:
                p = Process(target=self.episode, args=[env, i, render])
                p.start()
                l.append(p)
            else:
                reward = self.episode(env, i, render)
                evaluations.append(reward)
                neptune.log_metric('reward', reward)
            toc_episode = time.perf_counter()
            neptune.log_metric('episode_duration', toc_episode - tic_episode)
        if self.multi_processing:
            [p.join() for p in l]
        toc_training = time.perf_counter()
        neptune.log_metric('training_duration', toc_training - tic_training)
        np.save(f"./dqn/results/{file_name}", evaluations)

    def episode(self, env, nr_episode=0, render=False):
        state = env.reset()
        undiscounted_return = 0
        discount_factor = 0.99
        done = False
        time_step = 0
        while not done:
            if render:
                env.render()
            # 1. Select action according to policy
            action = self.policy(state)
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action)
            # 3. Integrate new experience into agent
            self.update(state, action, reward, next_state, done)
            state = next_state
            undiscounted_return += reward
            time_step += 1
        print(nr_episode, ":", undiscounted_return)
        return undiscounted_return

    """
     Overwrites target network weights with currently trained weights.
    """

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    """
     Selects a new action using epsilon-greedy exploration w.r.t. currently learned Q-Values.
    """

    def policy(self, state):
        Q_values = self.Q([state])[0]
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)

    """
     Predicts the currently learned Q-Values for a given batch of states.
    """

    def Q(self, states):
        return self.predict(np.array(states, dtype='float32'), self.policy_net)

    """
     Predicts the previously learned Q-Values for a given batch of states.
    """

    def target_Q(self, states):
        return self.predict(np.array(states, dtype='float32'), self.target_net)

    """
     Predicts the Q-Values of some model for a given batch of states.
    """

    def predict(self, states, model):
        return model.predict_on_batch(states)

    """
     Performs a learning update of the currently learned value function approximation.
    """

    def update(self, state, action, reward, next_state, done):
        self.memory.save((state, action, reward, next_state, done))
        self.warmup_phase = max(0, self.warmup_phase - 1)
        loss = None
        if self.warmup_phase == 0:
            self.training_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_linear_decay)
            minibatch = self.memory.sample_batch(self.minibatch_size)
            states, actions, rewards, next_states, dones = tuple(zip(*minibatch))
            current_Q_values = self.Q(states)
            next_Q_values = np.max(self.target_Q(next_states), axis=1)
            Q_targets = np.array(rewards) + self.gamma * next_Q_values
            for Q_values, action, reward, done, Q_target in zip(current_Q_values, actions, rewards, dones, Q_targets):
                if done:
                    Q_values[action] = reward
                else:
                    Q_values[action] = Q_target
            states = np.array(states, dtype='float32')
            training_targets = current_Q_values
            loss = self.policy_net.train_on_batch(states, training_targets)
            if self.training_count % self.target_update_interval == 0:
                self.update_target_network()
        return loss


def main():
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

