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
#from choose_strategy_functions import *
#from graph_generation import *

######################################################################
# Choose strategy function 
def choose_strategy_beliefs(tag, ingroup, outgroup, game, opponent_tag):
    if tag == opponent_tag:
        # Ingroup interaction

        choice_0_value = np.dot(game[0],
                                np.array([ingroup,
                                            1.0 - ingroup]))
        choice_1_value = np.dot(game[1],
                                np.array([ingroup,
                                            1.0 - ingroup]))
    else:
        # Outgroup interaction

        choice_0_value = np.dot(game[0],
                                np.array([outgroup,
                                            1.0 - outgroup]))
        choice_1_value = np.dot(game[1],
                                np.array([outgroup,
                                            1.0 - outgroup]))

    return 0 if choice_0_value > choice_1_value else 1

def choose_strategy_cooperate(tag, ingroup, outgroup, game, opponent_tag):
    return 0

def choose_strategy_defect(tag, ingroup, outgroup, game, opponent_tag):
    return 1

def choose_strategy_direct_action(tag, ingroup, outgroup, game, opponent_tag):

    if tag == opponent_tag:
        # Ingroup interaction

        strategy = np.random.choice([0,1], p=[ingroup, 1-ingroup])

    else:
        # Outgroup interaction

        strategy = np.random.choice([0,1], p=[outgroup, 1-outgroup])

    return strategy

choose_strategy_map = {
    "beliefs": choose_strategy_beliefs,
    "cooperate": choose_strategy_cooperate,
    "defect": choose_strategy_defect,
    "direct_action": choose_strategy_direct_action
}
###############################################################################
# Graph generation 
@jit(nopython=True)
def permute(matching, n):
    for i in range(n - 1):
        j = i + np.random.randint(n - i)
        temp = matching[i]
        matching[i] = matching[j]
        matching[j] = temp


def randomly_relabel(graph):
    permuted_nodes = list(graph.nodes)
    permute(permuted_nodes, len(permuted_nodes))
    relabel_dict = dict(zip(list(graph.nodes), permuted_nodes))
    new_edges = [(relabel_dict[source], relabel_dict[target]) for source, target in list(graph.edges())]

    graph.remove_edges_from(list(graph.edges()))
    graph.add_edges_from(new_edges)
    nx.relabel.relabel_nodes(graph, relabel_dict)

    return graph


def draw_graph_circular(graph, number_of_agents, number_of_0_tags):
    agent_0_colour = "#ff0000"  # Red
    agent_1_colour = "#0000ff"  # Blue
    nx.drawing.nx_pylab.draw_networkx(
        graph, pos=nx.drawing.layout.circular_layout(graph),
        nodelist=[i for i in range(number_of_agents)],
        node_color=[agent_0_colour if i < number_of_0_tags else
                    agent_1_colour for i in range(number_of_agents)],
        with_labels=True)
    plt.show()


def gerrymandered_graph(number_of_agents, number_of_0_tags, districts, rewire_amount):
    tags = np.ones(number_of_agents, dtype=int)

    for i in range(number_of_0_tags):
        tags[i] = 0

    district_assignment = [0 for _ in range(number_of_agents)]

    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(number_of_agents)])

    for current_agent in range(number_of_agents):
        if current_agent < number_of_0_tags:
            current_agent_tag = 0
        else:
            current_agent_tag = 1

        district = np.random.randint(0, len(districts[0]))
        while districts[current_agent_tag][district] == 0:
            district = np.random.randint(0, len(districts[0]))

        district_assignment[current_agent] = district
        districts[current_agent_tag][district] -= 1

    for focal_agent in range(number_of_agents):
        for other_agent in range(number_of_agents):
            if district_assignment[focal_agent] == district_assignment[other_agent] and focal_agent != other_agent:
                G.add_edge(focal_agent, other_agent)

    for _ in range(rewire_amount):
        edges = [edge for edge in G.edges]
        edge1 = edges[np.random.choice(range(len(edges)))]
        edge2 = edges[np.random.choice(range(len(edges)))]

        while (tags[edge1[1]] != tags[edge2[1]] or  # make sure target group is the same
               edge1[1] == edge2[1] or  # make sure targets are not the same
               edge1[0] == edge2[0] or  # make sure origins are not the same
               edge1[0] == edge2[1] or  # make sure target of edge2 is not origin of edge 1
               edge2[0] == edge1[1] or  # make sure target of edge1 is not origin of edge 2
               (edge1[0], edge2[1]) in edges or  # make sure swapped edges do not already exist
               (edge2[0], edge1[1]) in edges):
            edge1 = edges[np.random.choice(range(len(edges)))]
            edge2 = edges[np.random.choice(range(len(edges)))]

        G.remove_edge(*edge1)
        G.remove_edge(*edge2)
        G.add_edge(edge1[0], edge2[1])
        G.add_edge(edge2[0], edge1[1])

    return G


# Some districts for 24 agents. Source: https://doi.org/10.1038/s41586-019-1507-6
districts_1 = [[3, 3, 3, 3],
               [3, 3, 3, 3]]
districts_2 = [[5, 4, 2, 1],
               [1, 2, 4, 5]]
districts_3 = [[6, 2, 2, 2],
               [0, 4, 4, 4]]


def two_communities_graph(number_of_agents, initial_number_of_0_tags, rewire_amount, add_amount):
    tags = np.ones(number_of_agents, dtype=int)
    for i in range(initial_number_of_0_tags):
        tags[i] = 0

    G = nx.complete_graph(initial_number_of_0_tags)
    nx.to_directed(G)
    H = nx.complete_graph(list(range(initial_number_of_0_tags, number_of_agents)))
    nx.to_directed(H)

    F = nx.disjoint_union(G, H)

    for _ in range(rewire_amount):
        edges_F = [edge for edge in F.edges]
        edges_G = [edge for edge in G.edges]
        edges_H = [edge for edge in H.edges]
        edge1 = edges_G[np.random.choice(range(len(edges_G)))]
        edge2 = edges_H[np.random.choice(range(len(edges_H)))]

        while edge1 not in edges_F or \
            edge2 not in edges_F or \
            edge1[1] == edge2[1] or \
            edge1[0] == edge2[0] or \
            edge1[0] == edge2[1] or \
            edge2[0] == edge1[1] or \
            (edge1[0], edge2[1]) in edges_F or \
            (edge2[0], edge1[1]) in edges_F:
            edge1 = edges_G[np.random.choice(range(len(edges_G)))]
            edge2 = edges_H[np.random.choice(range(len(edges_H)))]

        F.remove_edge(*edge1)
        F.remove_edge(*edge2)
        F.add_edge(edge1[0], edge2[1])
        F.add_edge(edge2[0], edge1[1])

    for _ in range(add_amount):
        edges_F = [edge for edge in F.edges]
        nodes_G = [node for node in G.nodes]
        nodes_H = [node for node in H.nodes]
        node1 = nodes_G[np.random.choice(range(len(nodes_G)))]
        node2 = nodes_H[np.random.choice(range(len(nodes_H)))]

        while (node1, node2) in edges_F or \
            (node2, node1) in edges_F:
            node1 = nodes_G[np.random.choice(range(len(nodes_G)))]
            node2 = nodes_H[np.random.choice(range(len(nodes_H)))]

        direction = np.random.choice(2)
        if direction == 0:
            F.add_edge(node1, node2)
        else:
            F.add_edge(node2, node1)

    return F


def scale_free_graph(number_of_agents, initial_number_of_0_tags, alpha, beta, gamma):
    assert alpha + beta + gamma == 1

    tags = np.ones(number_of_agents, dtype=int)
    for i in range(initial_number_of_0_tags):
        tags[i] = 0

    G = nx.scale_free_graph(number_of_agents, alpha, beta, gamma)
    nx.to_directed(G)

    return G


graph_function_map = {
    "scale_free": lambda number_of_agents, number_of_0_tags: randomly_relabel(
        scale_free_graph(number_of_agents, number_of_0_tags, 0.41, 0.54, 0.05)),
    "two_communities": lambda number_of_agents, number_of_0_tags: two_communities_graph(number_of_agents,
                                                                                        number_of_0_tags,
                                                                                        number_of_agents // 2,
                                                                                        number_of_agents // 2),
    "gerrymandered": lambda number_of_agents, number_of_0_tags: gerrymandered_graph(number_of_agents, number_of_0_tags,
                                                                                    [[3, 3, 3, 3], [3, 3, 3, 3]],
                                                                                    number_of_agents // 2)
}

#############################################################################



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
        self.n_outputs = 1  # new belief

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.ReLU())
        # nn.Softmax(dim=-1))

    def predict(self, state):
        action = self.network(torch.FloatTensor(state))
        return action


class Policy():
    def __init__(self, agent, belief, gamma, epsilon, learning_rate=0.01):
        learning_rate = 0.01
        self.policy = Policy_Network()
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=learning_rate)

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
                 gamma=None, epsilon=None, learning_rate=None):
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
        self.ingroup_policy = Policy(self, "ingroup", gamma, epsilon, learning_rate)
        self.outgroup_policy = Policy(self, "outgroup", gamma, epsilon, learning_rate)


class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags, choose_strategy,
                 graph=None,
                 gamma=None, epsilon=None, learning_rate=0.01):

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
                  choose_strategy, gamma=gamma, epsilon=epsilon, learning_rate=learning_rate)
            for _ in range(self.number_of_agents)]

        for i in range(self.number_of_0_tags):
            self.agents[i].tag = 0
            self.agents[i].ingroup = tag0_initial_ingroup_belief
            self.agents[i].outgroup = tag0_initial_outgroup_belief

        self.matching_indices = list(range(self.number_of_agents))

        self.graph = graph

    def draw_graph(self):
        assert self.graph != None, "No graph available"
        agent_0_colour = "#ff0000"  # Red
        agent_1_colour = "#0000ff"  # Blue
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
            #np.random.shuffle(self.matching_indices)
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

        payoff_sum_0 = np.sum(np.exp(selection_intensity * payoff_0))
        payoff_sum_1 = np.sum(np.exp(selection_intensity * payoff_1))

        fitness_probabilities_0 = np.exp(selection_intensity * payoff_0) / payoff_sum_0
        fitness_probabilities_1 = np.exp(selection_intensity * payoff_1) / payoff_sum_1

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
    def compute_payoff_network(self):
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
        self.compute_payoff_network()

        new_agents = []

        for current_agent_index in range(self.number_of_agents):
            neighbourhood = [nbr for nbr in
                             self.graph.adj[current_agent_index].keys()]
            neighbourhood.append(current_agent_index)

            neighbourhood_payoffs = np.array([self.agents[i].payoff
                                              for i in neighbourhood])

            neighbourhood_fitness_sum = np.sum(np.exp(
                selection_intensity * neighbourhood_payoffs))

            neighbourhood_probabilities = np.exp(
                selection_intensity * neighbourhood_payoffs) / \
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
                current_agent_new_ingroup = self.agents[agent_to_imitate_index] \
                    .ingroup
                current_agent_new_outgroup = self.agents[agent_to_imitate_index] \
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
                number_of_total_steps = number_of_episodes*number_of_steps
                self.pre_training_RL(number_of_pretraining_episodes, number_of_pretraining_steps,
                                     rounds_per_step)
                for current_ep in range(number_of_episodes):
                    self.reinforce_RL(number_of_steps, rounds_per_step)
                    time_step += number_of_steps

                    if time_step % write_frequency == 0 \
                        or time_step == number_of_total_steps or time_step == number_of_total_steps - 1:
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

    ######## Reinforcement Learning Model on a Network #########

    def pre_training_RL_network(self, number_of_pretraining_episodes, number_of_pretraining_steps):
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
                self.compute_payoff_network()
                for i in range(self.number_of_agents):
                    self.agents[i].ingroup_policy.record_starting_state()
                    self.agents[i].outgroup_policy.record_starting_state()

                    # select action
                    self.agents[i].ingroup_policy.select_action_pretraining()
                    self.agents[i].outgroup_policy.select_action_pretraining()

                    # play a round of games to compute reward of new action
                    self.compute_payoff_network()

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

    def reinforce_RL_network(self, number_of_steps):
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
        self.compute_payoff_network()
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.record_starting_state()
            self.agents[i].outgroup_policy.record_starting_state()

        for _ in range(number_of_steps):
            for i in range(self.number_of_agents):
                # select action
                self.agents[i].ingroup_policy.select_action()
                self.agents[i].outgroup_policy.select_action()

                # play a round of games
                self.compute_payoff_network()

                # update reward and state
                self.agents[i].ingroup_policy.update_reward()
                self.agents[i].outgroup_policy.update_reward()

        # update policy
        for i in range(self.number_of_agents):
            self.agents[i].ingroup_policy.update_policy()
            self.agents[i].outgroup_policy.update_policy()

    def run_simulation_RL_network(self, random_seed, number_of_pretraining_episodes, number_of_pretraining_steps,
                       number_of_episodes, number_of_steps,
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

                time_step = 0
                number_of_total_steps = number_of_episodes*number_of_steps
                self.pre_training_RL_network(number_of_pretraining_episodes, number_of_pretraining_steps)
                for current_episode in range(number_of_episodes):
                    self.reinforce_RL_network(number_of_steps)
                    time_step += number_of_steps

                    if time_step % write_frequency == 0 \
                        or time_step == number_of_total_steps - 1 or time_step == number_of_total_steps:
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
            self.pre_training_RL_network(number_of_pretraining_episodes, number_of_pretraining_steps)
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
                self.reinforce_RL_network(number_of_steps)
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
        model = model_call(None, config["gamma"], config["epsilon"], config["learning_rate"])

        model.run_simulation_RL(config["random_seed"],
                                config["number_of_pretraining_episodes"],
                                config["number_of_pretraining_steps"],
                                config["number_of_episodes"],
                                config["number_of_steps"],
                                config["rounds_per_step"],
                                config["data_recording"],
                                config["data_file_path"],
                                config["write_frequency"])

    if config["model_type"] == 'RL_network':
        graph_func = graph_function_map[config["graph_type"]]
        graph = graph_func(config["number_of_agents"], config["initial_number_of_0_tags"])

        with open(config["graph_file_path"], 'w') as out_file:
            json.dump(nx.readwrite.json_graph.node_link_data(graph), out_file, indent=4)

        model = model_call(graph, config["gamma"], config["epsilon"], config["learning_rate"])
        model.run_simulation_RL_network(config["random_seed"],
                             config["number_of_pretraining_episodes"],
                             config["number_of_pretraining_steps"],
                             config["number_of_episodes"],
                             config["number_of_steps"],
                             config["data_recording"],
                             config["data_file_path"],
                             config["write_frequency"])


if __name__ == "__main__":
    main(sys.argv[1])
