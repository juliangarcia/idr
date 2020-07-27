import numpy as np
from numba import jit
import csv
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from choose_strategy_functions import *

@jit(nopython=True)
def permute(matching, n):
    for i in range(n-1):
        j = i + np.random.randint(n-i)
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

        # current state = [ current in/outgroup belief, current payoff aginst 0/1]
        self.current_state = [self.current_action, self.current_reward]

        # make updates
        self.states.append(self.current_state)
        self.rewards.append(self.current_reward)
        self.actions.append(self.current_action)

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

        # make updates
        self.states.append(self.current_state)
        self.rewards.append(self.current_reward)
        self.actions.append(self.current_action)

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
    def __init__(self, tag, ingroup, outgroup, choose_strategy, payoff=0.0):
        self.tag = tag
        self.ingroup = ingroup
        self.outgroup = outgroup
        self.payoff = payoff
        self.payoff_against_0 = 0.0
        self.payoff_against_1 = 0.0
        self.choose_strategy_func = choose_strategy
        self.choose_strategy = lambda *args: choose_strategy(
            self.tag, self.ingroup, self.outgroup, *args)
        self.ingroup_policy = Policy(self, "ingroup", 0.99, 0.05)
        self.outgroup_policy = Policy(self, "outgroup", 0.99, 0.05)

class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags, graph, choose_strategy):

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
                  choose_strategy)
            for _ in range(self.number_of_agents)]

        for i in range(self.number_of_0_tags):
            self.agents[i].tag = 0
            self.agents[i].ingroup = tag0_initial_ingroup_belief
            self.agents[i].outgroup = tag0_initial_outgroup_belief

        # Create graph
        self.graph = graph

    def draw_graph(self):
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

        return self.game[choice_focal, choice_other], \
               self.game[choice_other, choice_focal]

    def compute_payoff(self):
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

    def reinforce(self, number_of_steps):
        # this is a single episode so run multiple times (i.e. 100)

        # reset all agents states, rewards, actions sequences
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.states = []
            self.agents[i].ingroup_policy.rewards = []
            self.agents[i].ingroup_policy.actions = []
            self.agents[i].outgroup_policy.states = []
            self.agents[i].outgroup_policy.rewards = []
            self.agents[i].outgroup_policy.actions = []

        # include rewards from starting state, important!
        self.compute_payoff()
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.record_starting_state()
            self.agents[i].outgroup_policy.record_starting_state()

        for _ in range(number_of_steps):
            for i in range(self.number_of_agents):
                # select action
                self.agents[i].ingroup_policy.select_action()
                self.agents[i].outgroup_policy.select_action()

                # play a round of games
                self.compute_payoff()

                # update reward and state
                self.agents[i].ingroup_policy.update_reward()
                self.agents[i].outgroup_policy.update_reward()

        # update policy
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.update_policy()
            self.agents[i].outgroup_policy.update_policy()



    def run_simulation(self, random_seed, number_of_episodes, number_of_steps,
                       data_recording, data_file_path, write_frequency):

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)
                header_list = ["T"] + ["P" + str(i) for i in range(self.number_of_agents)] + \
                              ["I" + str(i) for i in range(self.number_of_agents)] + \
                              ["O" + str(i) for i in range(self.number_of_agents)]
                writer.writerow(header_list)

                for current_episode in range(number_of_episodes):
                    self.reinforce(number_of_steps)

                    if current_episode % write_frequency == 0 \
                            or current_episode == number_of_episodes - 1:
                        payoffs = np.array([agent.payoff
                                            for agent in self.agents])
                        ingroup = np.array([agent.ingroup
                                            for agent in self.agents])
                        outgroup = np.array([agent.outgroup
                                             for agent in self.agents])
                        writer.writerow(np.append([current_episode],
                                            np.append(payoffs,
                                                  np.append(ingroup,
                                                            outgroup))))
        else:
            for _ in range(number_of_episodes):
                self.reinforce(number_of_steps)


def main(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    with open(config["graph_file_path"]) as graph_file:
        graph = nx.readwrite.json_graph.node_link_graph(json.load(graph_file))
    model = Model(config["number_of_agents"],
                  config["R"], config["S"],
                  config["T"], config["P"],
                  config["tag0_initial_ingroup_belief"],
                  config["tag0_initial_outgroup_belief"],
                  config["tag1_initial_ingroup_belief"],
                  config["tag1_initial_outgroup_belief"],
                  config["initial_number_of_0_tags"],
                  graph,
                  choose_strategy_map[config["choose_strategy"]])
    model.run_simulation(config["random_seed"],
                         config["number_of_episodes"],
                         config["number_of_steps"],
                         config["data_recording"],
                         config["data_file_path"],
                         config["write_frequency"])

if __name__ == "__main__":
    #main(sys.argv[1])
    with open("graph.json") as graph_file:
        graph = nx.readwrite.json_graph.node_link_graph(json.load(graph_file))
    model = Model(24,
                  4, 1,
                  3, 2,
                  0.9,
                  0.9,
                  0.9,
                  0.9,
                  12,
                  graph,
                  choose_strategy_map["expected_payoff"])
    model.run_simulation(12345,
                         100,
                         1000,
                         False,
                         "data.csv", 10)
    print("Ingroup")
    for i in range(24):
        print(model.agents[i].ingroup)
    avg_in_0 = 0
    avg_in_1 = 0
    avg_total = 0
    for i in range(12):
        avg_in_0 += model.agents[i].ingroup
        avg_total += model.agents[i].ingroup
    for i in range(12):
        avg_in_1 += model.agents[i+12].ingroup
        avg_total += model.agents[i + 12].ingroup

    print("Average ingroup belief ", avg_total/24)
    print("Average tag 0 ingroup belief", avg_in_0/12)
    print("Average tag 1 ingroup belief", avg_in_1 / 12)

    print("Outgroup")
    for i in range(24):
        print(model.agents[i].outgroup)

    avg_in_0 = 0
    avg_in_1 = 0
    avg_total = 0
    for i in range(12):
        avg_in_0 += model.agents[i].outgroup
        avg_total += model.agents[i].outgroup
    for i in range(12):
        avg_in_1 += model.agents[i + 12].outgroup
        avg_total += model.agents[i + 12].outgroup

    print("Average outgroup belief ", avg_total / 24)
    print("Average tag 0 outgroup belief", avg_in_0 / 12)
    print("Average tag 1 outgroup belief", avg_in_1 / 12)

    model.draw_graph()
