import numpy
import random

class QLearner():

    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.action_counts = numpy.zeros(self.nr_actions)

    def Q(self, state):
        if state not in self.Q_values:
            self.Q_values[state] = numpy.zeros(self.nr_actions)
        return self.Q_values

    def policy(self, state):
        return epsilon_greedy(self.Q_values, self.action_counts, epsilon=self.epsilon)

    def update(self, state, action, reward, next_state):
        pass

def epsilon_greedy(Q_values, action_counts, epsilon=0.1):
    if numpy.random.rand() <= epsilon:
        return random.choice(range(len(Q_values)))
    else:
        return numpy.argmax(Q_values)
