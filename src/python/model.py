import numpy as np
from numba import jit
import csv
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from choose_strategy_functions import *
from graph_generation import *


# @jit(nopython=True)
# def permute(matching, n):
#     for i in range(n-1):
#         j = i + np.random.randint(n-i)
#         temp = matching[i]
#         matching[i] = matching[j]
#         matching[j] = temp

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
    def __init__(self, tag, ingroup, outgroup, choose_strategy, payoff=0.0,
                 gamma=None, epsilon=None):
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
                 initial_number_of_0_tags, choose_strategy,
                 graph=None,
                 gamma=None, epsilon=None):

        # 0 is cooperate
        # 1 is defect

        # Game payoff matrix
        self.game = np.array([[R, S], [T, P]])

        # let us assume pops are even
        assert number_of_agents % 2 == 0

        # could there be a loners option? non-interaction seems relevant
        self.number_of_agents = number_of_agents

        self.number_of_0_tags = initial_number_of_0_tags

        self.agents = [
            Agent(1,
                  tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                  choose_strategy, gamma=gamma, epsilon=epsilon)
            for _ in range(self.number_of_agents)]

        for i in range(self.number_of_0_tags):
            self.agents[i].tag = 0
            self.agents[i].ingroup = tag0_initial_ingroup_belief
            self.agents[i].outgroup = tag0_initial_outgroup_belief

        self.matching_indices = list(range(self.number_of_agents))

        self.graph = graph

    def draw_graph(self):
        assert self.graph != None, "No graph available"
        agent_0_colour = "#ff0000" # Red
        agent_1_colour = "#0000ff" # Blue
        nx.drawing.nx_pylab.draw_networkx(
            self.graph, pos=nx.drawing.layout.circular_layout(self.graph),
            nodelist=[i for i in range(self.number_of_agents)],
            node_color=[agent_0_colour if i < self.number_of_0_tags else
                        agent_1_colour for i in range(self.number_of_agents)],
            with_labels=True)
        plt.show()

    def encounter(self, agent_focal, agent_other):
        # assert 0 <= index_focal < self.number_of_agents
        # assert 0 <= index_other < self.number_of_agents
        # assert index_focal != index_other

        choice_focal = agent_focal.choose_strategy(self.game, agent_other.tag)
        choice_other = agent_other.choose_strategy(self.game, agent_focal.tag)

        return self.game[choice_focal, choice_other], self.game[choice_other, choice_focal]

    ############## Evolutionary Model on Well Mixed Population ###############

    def compute_payoff_evo(self, samples):
        for agent in self.agents:
            agent.payoff = 0.0

        for _ in range(samples):
            permute(self.matching_indices, self.number_of_agents)
            for i in range(0, self.number_of_agents, 2):
                focal_index = self.matching_indices[i]
                other_index = self.matching_indices[i + 1]
                payoff_focal, payoff_other = self.encounter(self.agents[focal_index], self.agents[other_index])
                self.agents[focal_index].payoff += payoff_focal
                self.agents[other_index].payoff += payoff_other
        
        for agent in self.agents:
            agent.payoff /= samples

    def step_evo(self, samples, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff_evo(samples)

        # Find the fitness probability distribution (using exponential selection intensity) for each group

        payoff_0 = np.array([agent.payoff for agent in self.agents[0:self.number_of_0_tags]])
        payoff_1 = np.array([agent.payoff for agent in self.agents[self.number_of_0_tags:]])

        payoff_sum_0 = np.sum(np.exp(selection_intensity*payoff_0))
        payoff_sum_1 = np.sum(np.exp(selection_intensity*payoff_1))

        fitness_probabilities_0 = np.exp(selection_intensity*payoff_0)/payoff_sum_0
        fitness_probabilities_1 = np.exp(selection_intensity*payoff_1)/payoff_sum_1

        new_agents = []

        # Create a new generation of agents.  Sampling occurs within
        # group only, to maintain group balance.
        for i in range(self.number_of_agents):
            if i < self.number_of_0_tags:
                current_agent_index = np.random.choice(range(self.number_of_0_tags),
                                                 p=fitness_probabilities_0)
            else:
                current_agent_index = np.random.choice(range(self.number_of_0_tags, self.number_of_agents),
                                                 p=fitness_probabilities_1)

            # Perturb agents belief
            if np.random.rand() <= perturbation_probability:
                current_agent_new_ingroup = np.random.normal(
                    self.agents[current_agent_index].ingroup, perturbation_scale)
                if current_agent_new_ingroup < 0:
                    current_agent_new_ingroup = 0.0
                if current_agent_new_ingroup > 1:
                    current_agent_new_ingroup = 1.0

                current_agent_new_outgroup = np.random.normal(
                    self.agents[current_agent_index].outgroup, perturbation_scale)
                if current_agent_new_outgroup < 0:
                    current_agent_new_outgroup = 0.0
                if current_agent_new_outgroup > 1:
                    current_agent_new_outgroup = 1.0
            else:
                current_agent_new_ingroup = self.agents[current_agent_index].ingroup
                current_agent_new_outgroup = self.agents[current_agent_index].outgroup

            new_agents.append(
                Agent(self.agents[current_agent_index].tag,
                      current_agent_new_ingroup, current_agent_new_outgroup,
                      self.agents[current_agent_index].choose_strategy_func,
                      payoff=self.agents[current_agent_index].payoff))

        self.agents = new_agents

    def run_simulation_evo(self, random_seed, number_of_steps, rounds_per_step,
                       selection_intensity, perturbation_probability,
                       perturbation_scale, data_recording,
                       data_file_path, write_frequency):

        np.random.seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)

                header_list = ["T"] + ["P" + str(i) for i in range(self.number_of_agents)] + \
                              ["I" + str(i) for i in range(self.number_of_agents)] + \
                              ["O" + str(i) for i in range(self.number_of_agents)]
                writer.writerow(header_list)

                for current_step in range(number_of_steps):
                    self.step_evo(rounds_per_step, selection_intensity,
                              perturbation_probability, perturbation_scale)

                    if current_step % write_frequency == 0 \
                            or current_step == number_of_steps - 1:
                        payoffs = np.array([agent.payoff
                                            for agent in self.agents])
                        ingroup = np.array([agent.ingroup
                                            for agent in self.agents])
                        outgroup = np.array([agent.outgroup
                                             for agent in self.agents])
                        writer.writerow(np.append([current_step],
                                            np.append(payoffs,
                                                  np.append(ingroup,
                                                            outgroup))))
        else:
            for _ in range(number_of_steps):
                self.step_evo(rounds_per_step, selection_intensity,
                          perturbation_probability, perturbation_scale)

    ################# Evolutionary on a Network #########################
    def compute_payoff_evo_network(self):
            # self.payoffs = np.zeros(self.number_of_agents)
            for agent in self.agents:
                agent.payoff = 0.0
                agent.payoff_against_0 = 0.0
                agent.payoff_against_1 = 0.0

            for focal_agent_index in range(self.number_of_agents):
                if len(self.graph.adj[focal_agent_index].keys()) > 0:

                    neighbours = [nbr for nbr in self.graph.adj[focal_agent_index].keys()]

                    games_played_against_0 = 0
                    games_played_against_1 = 0

                    for neighbour_index in neighbours:
                        payoff_focal, _ = self.encounter(self.agents[focal_agent_index], self.agents[neighbour_index])

                        self.agents[focal_agent_index].payoff += payoff_focal

                        if self.agents[neighbour_index].tag == 0:
                            games_played_against_0 += 1
                            self.agents[focal_agent_index].payoff_against_0 += payoff_focal
                        else:
                            games_played_against_1 += 1
                            self.agents[focal_agent_index].payoff_against_1 += payoff_focal

                    self.agents[focal_agent_index].payoff /= len(neighbours)
                    if games_played_against_0 > 0:
                        self.agents[focal_agent_index].payoff_against_0 /= games_played_against_0
                    if games_played_against_1 > 0:
                        self.agents[focal_agent_index].payoff_against_1 /= games_played_against_1

    def step_evo_network(self, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff_evo_network()

        new_agents = []

        for current_agent_index in range(self.number_of_agents):
            neighbourhood = [nbr for nbr in
                             self.graph.adj[current_agent_index].keys()]
            neighbourhood.append(current_agent_index)

            neighbourhood_payoffs = np.array([self.agents[i].payoff
                                              for i in neighbourhood])

            neighbourhood_fitness_sum = np.sum(np.exp(
                selection_intensity*neighbourhood_payoffs))

            neighbourhood_probabilities = np.exp(
                selection_intensity*neighbourhood_payoffs) / \
                neighbourhood_fitness_sum

            agent_to_imitate_index = np.random.choice(
                neighbourhood, p=neighbourhood_probabilities)

            if np.random.rand() <= perturbation_probability:
                current_agent_new_ingroup = np.random.normal(
                    self.agents[agent_to_imitate_index].ingroup,
                    perturbation_scale)
                if current_agent_new_ingroup < 0:
                    current_agent_new_ingroup = 0.0
                if current_agent_new_ingroup > 1:
                    current_agent_new_ingroup = 1.0

                current_agent_new_outgroup = np.random.normal(
                    self.agents[agent_to_imitate_index].outgroup,
                    perturbation_scale)
                if current_agent_new_outgroup < 0:
                    current_agent_new_outgroup = 0.0
                if current_agent_new_outgroup > 1:
                    current_agent_new_outgroup = 1.0
            else:
                current_agent_new_ingroup = self.agents[agent_to_imitate_index]\
                    .ingroup
                current_agent_new_outgroup = self.agents[agent_to_imitate_index]\
                    .outgroup

            new_agents.append(
                Agent(self.agents[current_agent_index].tag,
                      current_agent_new_ingroup, current_agent_new_outgroup,
                      self.agents[current_agent_index].choose_strategy_func,
                      payoff=self.agents[current_agent_index].payoff))

        self.agents = new_agents

    def run_simulation_evo_network(self, random_seed, number_of_steps,
                       selection_intensity, perturbation_probability,
                       perturbation_scale, data_recording,
                       data_file_path, write_frequency):

        np.random.seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)
                header_list = ["T"] + ["P" + str(i) for i in range(self.number_of_agents)] + \
                              ["I" + str(i) for i in range(self.number_of_agents)] + \
                              ["O" + str(i) for i in range(self.number_of_agents)]
                writer.writerow(header_list)

                for current_step in range(number_of_steps):
                    self.step_evo_network(selection_intensity,
                              perturbation_probability, perturbation_scale)

                    if current_step % write_frequency == 0 \
                            or current_step == number_of_steps - 1:
                        payoffs = np.array([agent.payoff
                                            for agent in self.agents])
                        ingroup = np.array([agent.ingroup
                                            for agent in self.agents])
                        outgroup = np.array([agent.outgroup
                                             for agent in self.agents])
                        writer.writerow(np.append([current_step],
                                            np.append(payoffs,
                                                  np.append(ingroup,
                                                            outgroup))))
        else:
            for _ in range(number_of_steps):
                self.step_evo_network(selection_intensity,
                          perturbation_probability, perturbation_scale)

    ######## Reinforcement Learning Model on Well Mixed Population #########

    def compute_payoff_RL(self, samples):
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

    def pre_training_RL(self, number_of_pretraining_episodes, number_of_pretraining_steps, rounds_per_step):
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
                self.compute_payoff_RL(rounds_per_step)
                for i in range(self.number_of_agents):
                    self.agents[i].ingroup_policy.record_starting_state()
                    self.agents[i].outgroup_policy.record_starting_state()

                    # select action
                    self.agents[i].ingroup_policy.select_action_pretraining()
                    self.agents[i].outgroup_policy.select_action_pretraining()

                    # play a round of games to compute reward of new action
                    self.compute_payoff_RL(rounds_per_step)

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

    def reinforce_RL(self, number_of_steps, rounds_per_step):
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
        self.compute_payoff_RL(rounds_per_step)
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.record_starting_state()
            self.agents[i].outgroup_policy.record_starting_state()

        for _ in range(number_of_steps):
            for i in range(self.number_of_agents):
                # select action
                self.agents[i].ingroup_policy.select_action()
                self.agents[i].outgroup_policy.select_action()

                # play a round of games
                self.compute_payoff_RL(rounds_per_step)

                # update reward and state
                self.agents[i].ingroup_policy.update_reward()
                self.agents[i].outgroup_policy.update_reward()

        # update policy
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.update_policy()
            self.agents[i].outgroup_policy.update_policy()

    def run_simulation_RL(self, random_seed, number_of_pretraining_episodes, number_of_pretraining_steps,
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
                self.pre_training_RL(number_of_pretraining_episodes, number_of_pretraining_steps,
                             rounds_per_step)
                for current_ep in range(number_of_episodes):
                    self.reinforce_RL(number_of_steps, rounds_per_step)
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
            self.pre_training_RL(number_of_pretraining_episodes, number_of_pretraining_steps,
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
                self.reinforce_RL(number_of_steps, rounds_per_step)
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
            # plot_results(ingroup_0, outgroup_0, ingroup_1, outgroup_1)
            return ingroup_0, outgroup_0, ingroup_1, outgroup_1

def main(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # Setup model call with standardised parameters
    model_call = lambda *args: Model(config["number_of_agents"],
                  config["R"], config["S"],
                  config["T"], config["P"],
                  config["tag0_initial_ingroup_belief"],
                  config["tag0_initial_outgroup_belief"],
                  config["tag1_initial_ingroup_belief"],
                  config["tag1_initial_outgroup_belief"],
                  config["initial_number_of_0_tags"],
                  choose_strategy_map[config["choose_strategy"]],
                  *args)

    if config["model_type"] == 'evo':
        model = model_call()
        model.run_simulation_evo(config["random_seed"], config["number_of_steps"],
                            config["rounds_per_step"],
                            config["selection_intensity"],
                            config["perturbation_probability"],
                            config["perturbation_scale"], config["data_recording"],
                            config["data_file_path"], config["write_frequency"])

    if config["model_type"] == 'evo_network':
        graph_func = graph_function_map[config["graph_type"]]
        graph = graph_func(config["number_of_agents"], config["initial_number_of_0_tags"])
        
        with open(config["graph_file_path"], 'w') as out_file:
            json.dump(nx.readwrite.json_graph.node_link_data(graph), out_file, indent=4)
        
        model = model_call(graph)
        model.run_simulation_evo_network(config["random_seed"], config["number_of_steps"],
                         config["selection_intensity"],
                         config["perturbation_probability"],
                         config["perturbation_scale"], config["data_recording"],
                         config["data_file_path"], config["write_frequency"])
    
    if config["model_type"] == 'RL':
        model = model_call(None, config["gamma"], config["epsilon"])
        model.run_simulation_RL(config["random_seed"],
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
