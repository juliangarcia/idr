import numpy as np
from numba import jit
import csv
import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class policy_estimator():
    def __init__(self):
        self.n_inputs = 2  # agent's belief
        self.n_outputs = 1 # new belief

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.ReLU())
            #nn.Softmax(dim=-1))

    def predict(self, state):
        action = self.network(torch.FloatTensor(state))
        return action

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    r = r -np.mean(r)
    return r

class Two_Agent_Model:
    def __init__(self, R, S, T, P, initial_belief, number_of_learning_agents, fixed_strategy):

        # 0 is cooperate
        # 1 is defect

        # Game payoff matrix
        self.game = np.array([[R, S], [T, P]])

        self.payoffs = np.zeros(2, dtype=float)
        self.fitnesses = np.zeros(2, dtype=float)

        self.belief = np.full(number_of_learning_agents, initial_belief, dtype=float)

        self.fixed_strategy = fixed_strategy
        self.number_of_learning_agents = number_of_learning_agents

        # Create policies for learning agents
        self.policies = {}
        for i in range(number_of_learning_agents):
            key = "policy_ingroup_agent_{}".format(str(i))
            policy = policy_estimator()
            self.policies[key] = policy


    def encounter(self, index_focal, index_other):

        if self.number_of_learning_agents == 2:

            # choice focal
            choice_0_value = np.dot(self.game[0], np.array([self.belief[index_focal],
                                                                  1.0 - self.belief[index_focal]]))
            choice_1_value = np.dot(self.game[1], np.array([self.belief[index_focal],
                                                                  1.0 - self.belief[index_focal]]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            # choice other
            choice_0_value = np.dot(self.game[0], np.array([self.belief[index_other],
                                                                  1.0 - self.belief[index_other]]))
            choice_1_value = np.dot(self.game[1], np.array([self.belief[index_other],
                                                                  1.0 - self.belief[index_other]]))
            choice_other = 0 if choice_0_value > choice_1_value else 1

            return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]

        else:
            # choice focal
            choice_0_value = np.dot(self.game[0], np.array([self.belief[index_focal],
                                                                  1.0 - self.belief[index_focal]]))
            choice_1_value = np.dot(self.game[1], np.array([self.belief[index_focal],
                                                                  1.0 - self.belief[index_focal]]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            choice_other = self.fixed_strategy

            return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]


    def compute_payoff(self, samples):
        self.payoffs = np.zeros(2)
        for _ in range(samples):
                focal_index = 0
                other_index = 1
                payoff_focal, payoff_other = self.encounter(focal_index, other_index)
                self.payoffs[focal_index] = self.payoffs[focal_index] + payoff_focal
                self.payoffs[other_index] = self.payoffs[other_index] + payoff_other
        self.payoffs = self.payoffs / samples

    def reinforce(self, policy_estimator, num_episodes, number_of_steps,
                  batch_size, gamma):
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 1
        beliefs = []
        beliefs.append(self.belief[0])

        # Define optimizer
        optimizer = optim.Adam(policy_estimator.network.parameters(),
                               lr=0.01)
        ep = 0
        while ep < num_episodes:
            states = []
            rewards = []
            actions = []
            done = False
            for _ in range(number_of_steps):
                # Play a round of games
                self.compute_payoff(batch_size)
                current_state = [self.belief[0], self.payoffs[0]]
                # Select action
                action = policy_estimator.predict(
                    current_state).detach().numpy()
                if action > 1:
                    action = 1.0
                if action < 0:
                    action = 0.0

                # Calculate reward
                self.belief[0] = action
                p1,p2 = self.encounter(0,1)
                r = p1

                states.append(current_state)
                rewards.append(r)
                actions.append(action)

            # update policy
            batch_rewards.extend(discount_rewards(rewards, gamma))
            batch_states.extend(states)
            batch_actions.extend(actions)
            total_rewards.append(sum(rewards))

            optimizer.zero_grad()
            state_tensor = torch.FloatTensor(batch_states)
            reward_tensor = torch.FloatTensor(batch_rewards)

            # Actions are used as indices, must be LongTensor
            action_tensor = torch.LongTensor(
                batch_actions)

            # Calculate loss
            logprob = torch.log(
                policy_estimator.predict(state_tensor))

            if len(action_tensor.size()) == 1:
                # self.belief = 1
                break
            selected_logprobs = reward_tensor * \
                                torch.gather(logprob, 1,
                                             action_tensor).squeeze()
            loss = -selected_logprobs.mean()

            # Calculate gradients
            loss.backward()
            # Apply gradients
            optimizer.step()

            batch_rewards = []
            batch_actions = []
            batch_states = []

            avg_rewards = np.mean(total_rewards[-100:])
            # Print running average
            #print("\rEp: " + "{}" + " Average of last 100:" +
            #      "{:.2f}".format(
            #          ep + 1, avg_rewards), end="")

            if (ep+1) % 10 == 0:
                print('\rEpisode {}\tAverage reward of last 100: {:.2f}'.format(ep+1, avg_rewards))
                print("Current belief ", self.belief[0])
            ep += 1
            beliefs.append(self.belief[0])

        actual_rewards = np.array(total_rewards)/number_of_steps
        return actual_rewards, beliefs

def plot_results(rewards, beliefs):
    window = int(len(rewards)/ 20)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=[9, 9]);
    rolling_mean = pd.Series(rewards).rolling(window).mean()
    std = pd.Series(rewards).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(rewards)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title('Rewards'.format(window))
    ax1.set_xlabel('Episode');
    ax1.set_ylabel('Rewards')

    ax2.plot(beliefs)
    ax2.set_title('Beliefs')
    ax2.set_xlabel('Episode');
    ax2.set_ylabel('Beliefs')

    fig.tight_layout(pad=2)
    plt.show()

if __name__ == "__main__":
    model = Two_Agent_Model(4, 1, 3, 2, 0.9, 1, 1)
    policy_est = policy_estimator()
    rewards, beliefs = model.reinforce(policy_est, 100, 1000, 100, 0.99)
    plot_results(rewards, beliefs)

    print(model.belief)

'''
def main(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    model = Model(config["number_of_agents"],
                  config["R"], config["S"],
                  config["T"], config["P"],
                  config["tag0_initial_ingroup_belief"],
                  config["tag0_initial_outgroup_belief"],
                  config["tag1_initial_ingroup_belief"],
                  config["tag1_initial_outgroup_belief"],
                  config["initial_number_of_0_tags"])
    model.run_simulation(config["random_seed"], config["number_of_steps"],
                         config["rounds_per_step"],
                         config["selection_intensity"],
                         config["perturbation_probability"],
                         config["perturbation_scale"], config["data_recording"],
                         config["data_file_path"], config["write_frequency"])


if __name__ == "__main__":
    main(sys.argv[1])
'''
