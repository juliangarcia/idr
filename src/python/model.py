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
from choose_strategy_functions import *

@jit(nopython=True)
def permute(matching, n):
    for i in range(n - 1):
        j = i + np.random.randint(n - i)
        temp = matching[i]
        matching[i] = matching[j]
        matching[j] = temp

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

class Policy():
    def __init__(self, agent, belief, gamma, epsilon):
        self.policy = Policy_Network()
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=0.01)

        self.gamma = gamma
        self.epsilon = epsilon
        self.agent = agent
        self.belief = belief

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

        self.initial_belief = self.current_belief

        self.avg_rewards = 0

        self.states = []
        self.rewards = []
        self.actions = []

        self.current_action = self.current_belief
        self.current_reward = 0
        self.current_state = 0

    def select_action(self):
        # Select action
        predicted_action = self.policy.predict(
            self.current_state).detach().numpy()
        self.current_action = predicted_action[0]
        if self.current_action >= 1:
            self.current_action = 0.99
        if self.current_action <= 0:
            self.current_action = 0.01

        # introduce noise
        x = np.random.uniform()
        if x < self.epsilon:
            experimental_action = np.random.uniform(size=1)
            self.current_action = experimental_action[0]

    def select_action_pretraining(self):
        if self.belief == "ingroup":
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_0
            else:
                self.current_reward = self.agent.payoff_against_1
        else:
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_1
            else:
                self.current_reward = self.agent.payoff_against_0

        self.current_state = [self.current_belief, self.current_reward]
        # Select action
        predicted_action = self.policy.predict(
            self.current_state).detach().numpy()
        self.current_action = predicted_action[0]
        if self.current_action >= 1:
            self.current_action = 0.99
        if self.current_action <= 0:
            self.current_action = 0.01

    def update_reward(self):
        if self.belief == "ingroup":
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_0
            else:
                self.current_reward = self.agent.payoff_against_1
        else:
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_1
            else:
                self.current_reward = self.agent.payoff_against_0

        # make updates
        self.states.append(self.current_state)
        self.rewards.append(self.current_reward)
        self.actions.append(self.current_action)

        # update state
        self.current_state = [self.current_action, self.current_reward]

        # update agent class
        if self.belief == "ingroup":
            self.agent.ingroup = self.current_action
        else:
            self.agent.outgroup = self.current_action

        self.current_belief = self.current_action

    def record_starting_state(self):
        if self.belief == "ingroup":
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_0
            else:
                self.current_reward = self.agent.payoff_against_1
        else:
            if self.agent.tag == 0:
                self.current_reward = self.agent.payoff_against_1
            else:
                self.current_reward = self.agent.payoff_against_0

        self.current_state = [self.current_belief, self.current_reward]
        self.current_action = self.current_belief

    def discount_rewards(self, rewards):
        r = np.array([self.gamma ** i * rewards[i]
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        r = r - np.mean(r)
        return r

    def update_policy(self):
        self.batch_rewards.extend(self.discount_rewards(self.rewards))
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

        selected_logprobs = reward_tensor * \
                            torch.gather(logprob, 1,
                                         action_tensor.unsqueeze(1)).squeeze()
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()

        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []

        self.avg_rewards = np.mean(self.total_rewards[-100:])

        self.beliefs.append(self.current_belief)

class Agent:
    def __init__(self, tag, ingroup, outgroup, choose_strategy, gamma, epsilon, payoff=0.0):
        self.tag = tag
        self.ingroup = ingroup
        self.outgroup = outgroup
        self.initial_ingroup = ingroup
        self.initial_outgroup = outgroup
        self.payoff = payoff
        self.payoff_against_0 = 0.0
        self.payoff_against_1 = 0.0
        self.choose_strategy_func = choose_strategy
        self.choose_strategy = lambda *args: choose_strategy(
            self.tag, self.ingroup, self.outgroup, *args)
        self.ingroup_policy = Policy(self, "ingroup", gamma, epsilon)
        self.outgroup_policy = Policy(self, "outgroup", gamma, epsilon)


class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags, choose_strategy, gamma, epsilon):

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

        self.agents = [
            Agent(1,
                  tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                  choose_strategy, gamma, epsilon)
            for _ in range(self.number_of_agents)]

        for i in range(self.number_of_0_tags):
            self.agents[i].tag = 0
            self.agents[i].ingroup = tag0_initial_ingroup_belief
            self.agents[i].outgroup = tag0_initial_outgroup_belief

        self.matching_indices = list(range(self.number_of_agents))

    def encounter(self, agent_focal, agent_other):
        # assert 0 <= index_focal < self.number_of_agents
        # assert 0 <= index_other < self.number_of_agents
        # assert index_focal != index_other

        choice_focal = agent_focal.choose_strategy(self.game, agent_other.tag)
        choice_other = agent_other.choose_strategy(self.game, agent_focal.tag)

        return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]

    def compute_payoff(self, samples):
        # self.payoffs = np.zeros(self.number_of_agents)
        for agent in self.agents:
            agent.payoff = 0.0
            agent.payoff_against_0 = 0.0
            agent.payoff_against_1 = 0.0

        games_against_0 = np.zeros(self.number_of_agents)
        games_against_1 = np.zeros(self.number_of_agents)

        for _ in range(samples):
            #np.random.shuffle(self.matching_indices)
            permute(self.matching_indices, self.number_of_agents)
            for i in range(0, self.number_of_agents, 2):
                focal_index = self.matching_indices[i]
                other_index = self.matching_indices[i + 1]
                payoff_focal, payoff_other = self.encounter(self.agents[focal_index], self.agents[other_index])
                self.agents[focal_index].payoff += payoff_focal
                self.agents[other_index].payoff += payoff_other

                if self.agents[focal_index].tag == 0:
                    if self.agents[other_index].tag == 0:
                        games_against_0[focal_index] += 1
                        games_against_0[other_index] += 1
                        self.agents[focal_index].payoff_against_0 += payoff_focal
                        self.agents[other_index].payoff_against_0 += payoff_other
                    else:
                        games_against_1[focal_index] += 1
                        games_against_0[other_index] += 1
                        self.agents[focal_index].payoff_against_1 += payoff_focal
                        self.agents[other_index].payoff_against_0 += payoff_other
                else:
                    if self.agents[other_index].tag == 0:
                        games_against_0[focal_index] += 1
                        games_against_1[other_index] += 1
                        self.agents[focal_index].payoff_against_0 += payoff_focal
                        self.agents[other_index].payoff_against_1 += payoff_other
                    else:
                        games_against_1[focal_index] += 1
                        games_against_1[other_index] += 1
                        self.agents[focal_index].payoff_against_1 += payoff_focal
                        self.agents[other_index].payoff_against_1 += payoff_other

        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.payoff /= samples
            if games_against_0[i] != 0:
                agent.payoff_against_0 /= games_against_0[i]
            if games_against_1[i] != 0:
                agent.payoff_against_1 /= games_against_1[i]


    def pre_training(self, number_of_pretraining_episodes, number_of_pretraining_steps, rounds_per_step):
        for ep in range(number_of_pretraining_episodes):
            # reset all agents states, rewards, actions sequences
            for i in range(self.number_of_agents):
                self.agents[i].ingroup_policy.states = []
                self.agents[i].ingroup_policy.rewards = []
                self.agents[i].ingroup_policy.actions = []
                self.agents[i].outgroup_policy.states = []
                self.agents[i].outgroup_policy.rewards = []
                self.agents[i].outgroup_policy.actions = []

            for _ in range(number_of_pretraining_steps):
                self.compute_payoff(rounds_per_step)
                for i in range(self.number_of_agents):
                    self.agents[i].ingroup_policy.record_starting_state()
                    self.agents[i].outgroup_policy.record_starting_state()

                    # select action
                    self.agents[i].ingroup_policy.select_action_pretraining()
                    self.agents[i].outgroup_policy.select_action_pretraining()

                    # play a round of games to compute reward of new action
                    self.compute_payoff(rounds_per_step)

                    # update reward and state
                    self.agents[i].ingroup_policy.update_reward()
                    self.agents[i].outgroup_policy.update_reward()

                    # reset belief to initial belief (pretraining only)
                    self.agents[i].ingroup = self.agents[i].initial_ingroup
                    self.agents[i].outgroup = self.agents[i].initial_outgroup
                    self.agents[i].ingroup_policy.current_belief = self.agents[i].initial_ingroup
                    self.agents[i].outgroup_policy.current_belief = self.agents[i].initial_outgroup

            # update policy
            for i in range(self.number_of_agents):
                self.agents[i].ingroup_policy.update_policy()
                self.agents[i].outgroup_policy.update_policy()


    def reinforce(self, number_of_steps, rounds_per_step):
        # this is a single episode so run multiple times (i.e. 100)

        # reset all agents states, rewards, actions sequences
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.states = []
            self.agents[i].ingroup_policy.rewards = []
            self.agents[i].ingroup_policy.actions = []
            self.agents[i].outgroup_policy.states = []
            self.agents[i].outgroup_policy.rewards = []
            self.agents[i].outgroup_policy.actions = []

        # play a starting round of games
        self.compute_payoff(rounds_per_step)
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.record_starting_state()
            self.agents[i].outgroup_policy.record_starting_state()

        for _ in range(number_of_steps):
            for i in range(self.number_of_agents):
                # select action
                self.agents[i].ingroup_policy.select_action()
                self.agents[i].outgroup_policy.select_action()

                # play a round of games
                self.compute_payoff(rounds_per_step)

                # update reward and state
                self.agents[i].ingroup_policy.update_reward()
                self.agents[i].outgroup_policy.update_reward()

        # update policy
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.update_policy()
            self.agents[i].outgroup_policy.update_policy()

    def run_simulation(self, random_seed, number_of_pretraining_episodes, number_of_pretraining_steps,
                       number_of_episodes, number_of_steps,
                       rounds_per_step, data_recording, data_file_path,
                       write_frequency):

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)

                header_list = ["T"] + ["P" + str(i) for i in range(self.number_of_agents)] + \
                              ["I" + str(i) for i in range(self.number_of_agents)] + \
                              ["O" + str(i) for i in range(self.number_of_agents)]
                writer.writerow(header_list)

                time_step = 0
                self.pre_training(number_of_pretraining_episodes, number_of_pretraining_steps,
                             rounds_per_step)
                for current_ep in range(number_of_episodes):
                    self.reinforce(number_of_steps, rounds_per_step)
                    time_step += number_of_steps

                    if current_ep % write_frequency == 0 \
                        or current_ep == number_of_episodes - 1:
                        payoffs = np.array([agent.payoff
                                            for agent in self.agents])
                        ingroup = np.array([agent.ingroup
                                            for agent in self.agents])
                        outgroup = np.array([agent.outgroup
                                             for agent in self.agents])
                        writer.writerow(np.append([time_step],
                                                  np.append(payoffs,
                                                            np.append(ingroup,
                                                                      outgroup))))
        else:
            self.pre_training(number_of_pretraining_episodes, number_of_pretraining_steps,
                         rounds_per_step)
            print("Finsihed pre-training")

            # create arrays to store the average in/outgroup beliefs of the tag groups
            ingroup_0 = np.zeros(number_of_episodes + 1, dtype=float)
            ingroup_1 = np.zeros(number_of_episodes + 1, dtype=float)
            outgroup_0 = np.zeros(number_of_episodes + 1, dtype=float)
            outgroup_1 = np.zeros(number_of_episodes + 1, dtype=float)

            # include initial beliefs
            for i in range(self.number_of_0_tags):
                ingroup_0[0] += self.agents[i].ingroup
                outgroup_0[0] += self.agents[i].outgroup
            for i in range(self.number_of_0_tags, self.number_of_agents):
                ingroup_1[0] += self.agents[i].ingroup
                outgroup_1[0] += self.agents[i].outgroup

            # run the model and after each episode record the average beliefs
            for j in range(number_of_episodes):
                self.reinforce(number_of_steps, rounds_per_step)
                print("Episode ", str(j + 1))

                for i in range(self.number_of_0_tags):
                    ingroup_0[j + 1] += self.agents[i].ingroup
                    outgroup_0[j + 1] += self.agents[i].outgroup
                for i in range(self.number_of_0_tags, self.number_of_agents):
                    ingroup_1[j + 1] += self.agents[i].ingroup
                    outgroup_1[j + 1] += self.agents[i].outgroup

            ingroup_0 = ingroup_0 / self.number_of_0_tags
            outgroup_0 = outgroup_0 / self.number_of_0_tags
            ingroup_1 = ingroup_1 / (self.number_of_agents - self.number_of_0_tags)
            outgroup_1 = outgroup_1 / (self.number_of_agents - self.number_of_0_tags)
            plot_results(ingroup_0, outgroup_0, ingroup_1, outgroup_1)
            return ingroup_0, outgroup_0, ingroup_1, outgroup_1

def plot_results(ingroup_0, outgroup_0, ingroup_1, outgroup_1):
    #window = int(len(rewards)/ 20)
    window = 1
    fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(4, 1, figsize=[9, 9]);
    rolling_mean = pd.Series(ingroup_0).rolling(window).mean()
    std = pd.Series(ingroup_0).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(ingroup_0)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title('Average Ingroup belief of tag 0'.format(window))
    ax1.set_xlabel('Episode');
    ax1.set_ylabel('Belief')
    ax1.set_ylim([0,1])

    rolling_mean2 = pd.Series(outgroup_0).rolling(window).mean()
    std2 = pd.Series(outgroup_0).rolling(window).std()
    ax2.plot(rolling_mean2)
    ax2.fill_between(range(len(outgroup_0)), rolling_mean2 - std2, rolling_mean2 + std2, color='orange', alpha=0.2)
    ax2.set_title('Average Outgroup belief of tag 0'.format(window))
    ax2.set_xlabel('Episode');
    ax2.set_ylabel('Belief')
    ax2.set_ylim([0, 1])

    rolling_mean3 = pd.Series(ingroup_1).rolling(window).mean()
    std3 = pd.Series(ingroup_1).rolling(window).std()
    ax3.plot(rolling_mean3)
    ax3.fill_between(range(len(ingroup_1)), rolling_mean3 - std3, rolling_mean3 + std3, color='orange', alpha=0.2)
    ax3.set_title('Average Ingroup belief of tag 1'.format(window))
    ax3.set_xlabel('Episode');
    ax3.set_ylabel('Belief')
    ax3.set_ylim([0, 1])

    rolling_mean4 = pd.Series(outgroup_1).rolling(window).mean()
    std4 = pd.Series(outgroup_1).rolling(window).std()
    ax4.plot(rolling_mean4)
    ax4.fill_between(range(len(outgroup_1)), rolling_mean4 - std4, rolling_mean4 + std4, color='orange', alpha=0.2)
    ax4.set_title('Average Outgroup belief of tag 1'.format(window))
    ax4.set_xlabel('Episode');
    ax4.set_ylabel('Belief')
    ax4.set_ylim([0, 1])

    fig.tight_layout(pad=2)
    plt.show()

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
                  config["initial_number_of_0_tags"],
                  choose_strategy_map[config["choose_strategy"]],
                  config["gamma"],
                  config["epsilon"])
    model.run_simulation(config["random_seed"],
                         config["number_of_pretraining_episodes"],
                         config["number_of_pretraining_steps"],
                         config["number_of_episodes"],
                         config["number_of_steps"],
                         config["rounds_per_step"],
                         config["data_recording"],
                         config["data_file_path"],
                         config["write_frequency"])

if __name__ == "__main__":
    main(sys.argv[1])
