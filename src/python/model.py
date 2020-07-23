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

class Policy_Network():
    def __init__(self):
        self.n_inputs = 2  # agent's belief, payoff
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

class Policy()
    def __init__(self, agent, belief, gamma):
        policy = Policy_Network()
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=0.01)

        self.gamma = 0.99
        self.agent = agent

        self.total_rewards = []
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []

        if belief == "ingroup":
            self.beliefs = [self.agent.ingroup]
            self.current_belief = self.agent.ingroup
        else:
            self.beliefs = [self.agent.outgroup]
            self.current_belief = self.agent.outgroup

        self.avg_rewards = 0

        self.states = []
        self.rewards = []
        self.actions = []

        self.current_action = 0
        self.current_reward = 0
        self.current_state = 0

    def select_action(self):
        self.current_state = [self.current_belief, self.agent.payoff]
        # Select action
        self.current_action = self.policy.predict(
            self.current_state).detach().numpy()
        if self.current_action > 1:
            self.current_action = 1.0
        if self.current_action < 0:
            self.current_action = 0.0

    def update_reward(self, reward):
        self.current_reward = reward

        # make updates
        self.states.append(self.current_state)
        self.rewards.append(self.current_rewards)
        self.actions.append(self.current_action)
        if belief == "ingroup":
            self.agent.ingroup = self.current_action
        else:
            self.agent.outgroup = self.current_action

        self.current_belief = self.current_action

    def discount_rewards(self, rewards):
        r = np.array([self.gamma ** i * rewards[i]
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        r = r - np.mean(r)
        return r

    def update_policy(self):
        self.batch_rewards.extend(discount_rewards(self.rewards, self.gamma))
        self.batch_states.extend(self.states)
        self.batch_actions.extend(self.actions)
        self.total_rewards.append(sum(self.rewards))

        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(self.batch_states)
        reward_tensor = torch.FloatTensor(self.batch_rewards)

        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(self.batch_actions)

        # Calculate loss
        logprob = torch.log(self.policy.predict(state_tensor))

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
        self.optimizer.step()

        self.batch_rewards[i] = []
        self.batch_actions[i] = []
        self.batch_states[i] = []

        self.avg_rewards = np.mean(self.total_rewards[-100:])

        self.beliefs.append(self.current_belief)


class Agent:
    def __init__(self, id, ingroup_belief, outgroup_belief, tag):
    self.id = id
    self.ingroup = ingroup_belief
    self.outgroup = outgroup_belief
    self.tag = tag
    self.payoff = 0
    self.ingroup_policy = Policy(self, "ingroup")
    self.outgroup_policy = Policy(self, "outgroup")


class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags):

        # 0 is cooperate
        # 1 is defect

        # Game payoff matrix
        self.game = np.array([[R, S], [T, P]])

        # let us assume pops are even
        assert number_of_agents % 2 == 0

        # could there be a loners option? non-interaction seems relevant
        self.number_of_agents = number_of_agents

        self.number_of_0_tags = initial_number_of_0_tags

        # these contain probabilities that the in-out group
        # will play strategy 0 - cooperate

        self.tags = np.ones(number_of_agents, dtype=int)
        self.ingroup = np.full(number_of_agents, tag1_initial_ingroup_belief, dtype=float)
        self.outgroup = np.full(number_of_agents, tag1_initial_outgroup_belief, dtype=float)

        for i in range(self.number_of_0_tags):
            self.tags[i] = 0
            self.ingroup[i] = tag0_initial_ingroup_belief
            self.outgroup[i] = tag0_initial_outgroup_belief

        self.matching_indices = list(range(self.number_of_agents))
        self.payoffs = np.zeros(number_of_agents, dtype=float)

        self.agents = []
        for i in range(self.number_of_agents):
            self.agents.append(Agent(i, self.ingroup[i], self.outgroup[i], self.tag[i]))

    def encounter(self, index_focal, index_other):
        assert 0 <= index_focal < self.number_of_agents
        assert 0 <= index_other < self.number_of_agents
        assert index_focal != index_other

        if self.tags[index_focal] == self.tags[index_other]:
            # ingroup interaction

            # choice focal
            choice_0_value = np.dot(self.game[0], np.array([self.ingroup[index_focal],
                                                                  1.0 - self.ingroup[index_focal]]))
            choice_1_value = np.dot(self.game[1], np.array([self.ingroup[index_focal],
                                                                  1.0 - self.ingroup[index_focal]]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            # choice other
            choice_0_value = np.dot(self.game[0], np.array([self.ingroup[index_other],
                                                                  1.0 - self.ingroup[index_other]]))
            choice_1_value = np.dot(self.game[1], np.array([self.ingroup[index_other],
                                                                  1.0 - self.ingroup[index_other]]))
            choice_other = 0 if choice_0_value > choice_1_value else 1

            return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]

        else:
            # outgroup interaction

            # choice focal
            choice_0_value = np.dot(self.game[0], np.array([self.outgroup[index_focal],
                                                                  1.0 - self.outgroup[index_focal]]))
            choice_1_value = np.dot(self.game[1], np.array([self.outgroup[index_focal],
                                                                  1.0 - self.outgroup[index_focal]]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            # choice other
            choice_0_value = np.dot(self.game[0], np.array([self.outgroup[index_other],
                                                                  1.0 - self.outgroup[index_other]]))
            choice_1_value = np.dot(self.game[1], np.array([self.outgroup[index_other],
                                                                  1.0 - self.outgroup[index_other]]))
            choice_other = 0 if choice_0_value > choice_1_value else 1

            return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]

    def compute_payoff(self, samples):
        self.payoffs = np.zeros(self.number_of_agents)
        for _ in range(samples):
            permute(self.matching_indices, self.number_of_agents)
            for i in range(0, self.number_of_agents, 2):
                focal_index = self.matching_indices[i]
                other_index = self.matching_indices[i + 1]
                payoff_focal, payoff_other = self.encounter(focal_index, other_index)
                self.payoffs[focal_index] = self.payoffs[focal_index] + payoff_focal
                self.payoffs[other_index] = self.payoffs[other_index] + payoff_other
        self.payoffs = self.payoffs/samples
        for i in range(self.number_of_agents):
            self.agents[i].payoff = self.payoffs[i]

    def compute_individual_payoff(self, agent, samples):
        # plays an agent against other agents but does not change self.payoffs
        #array and returns that agents payoff
        agent_payoff = 0
        players = np.random.randint(low=1, high=self.number_of_agents, size=samples)
        while agent.id in players:
            players = np.random.randint(low=1, high=self.number_of_agents, size=samples)
        for i in players:
            payoff_focal, payoff_other = self.encounter(agent.id, i)
            agent_payoff += payoff_focal
        return agent_payoff/samples

    def reinforce(self, random_seed, number_of_episodes, number_of_steps,
                  rounds_per_step):
        for ep in range(number_of_episodes):
            # reset all agents states, rewards, actions sequences
            for i in range(self.number_of_agents):
                self.agents[i].ingroup_policy.states = []
                self.agents[i].ingroup_policy.rewards = []
                self.agents[i].ingroup_policy.actions = []
                self.agents[i].outgroup_policy.states = []
                self.agents[i].outgroup_policy.rewards = []
                self.agents[i].outgroup_policy.actions = []

            for _ in range(number_of_steps):
                # Play a round of games
                self.compute_payoff(rounds_per_step)

                for i in range(self.number_of_agents):
                    # select action
                    self.agents[i].ingroup_policy.select_action()
                    self.agents[i].outgroup_policy.select_action()

                    # pdate reward and state
                    self.agents[i].ingroup_policy.update_reward(
                        self.compute_individual_payoff(self.agents[i]))
                    self.agents[i].outgroup_policy.update_reward(
                        self.compute_individual_payoff(self.agents[i]))

            # update policy
            for i in range(self.number_of_agents):
                self.agents[i].ingroup_policy.update_policy()
                self.agents[i].outgroup_policy.update_policy()

            if (ep+1) % 10 == 0:
                print("Episode: ", str(ep+1))

    def plot_results(self, rewards, beliefs):
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

    def run_simulation(self, random_seed, number_of_episodes, number_of_steps,
                       rounds_per_step, data_recording, data_file_path,
                       write_frequency):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.reinforce(random_seed, number_of_episodes, number_of_steps,
                  rounds_per_step)

        # data recording

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
    model.run_simultaion(config["random_seed"],
                        config["number_of_episodes"],
                        config["number_of_steps"],
                        config["rounds_per_step"],
                        config["data_recording"],
                        config["data_file_path"],
                        config["write_frequency"])

if __name__ == "__main__":
    main(sys.argv[1])