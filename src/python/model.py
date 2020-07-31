import numpy as np
from numba import jit
import csv
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
from choose_strategy_functions import *
from graph_generation import *


# @jit(nopython=True)
# def permute(matching, n):
#     for i in range(n-1):
#         j = i + np.random.randint(n-i)
#         temp = matching[i]
#         matching[i] = matching[j]
#         matching[j] = temp

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

class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags, choose_strategy,
                 graph=None):

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
                  choose_strategy)
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
                      self.agents[current_agent_index].payoff))

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


if __name__ == "__main__":
    main(sys.argv[1])
