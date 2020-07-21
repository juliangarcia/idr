import numpy as np
from numba import jit
import csv
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt

@jit(nopython=True)
def permute(matching, n):
    for i in range(n-1):
        j = i + np.random.randint(n-i)
        temp = matching[i]
        matching[i] = matching[j]
        matching[j] = temp


class Model:
    def __init__(self, number_of_agents, R, S, T, P,
                 tag0_initial_ingroup_belief, tag0_initial_outgroup_belief,
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief,
                 initial_number_of_0_tags, graph):

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

        # Create graph
        self.graph = graph

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

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
            game_played = [False for _ in range(self.number_of_agents)]
            permute(self.matching_indices, self.number_of_agents)
            for focal_index in self.matching_indices:
                if game_played[focal_index]:
                    continue
                viable_neighbours = [nbr for nbr in
                    self.graph.adj[focal_index].keys() if not game_played[nbr]]
                if len(viable_neighbours) > 0:
                    other_index = np.random.choice(viable_neighbours)
                    game_played[focal_index] = True
                    game_played[other_index] = True
                    payoff_focal, payoff_other = self.encounter(focal_index,
                                                                other_index)
                    self.payoffs[focal_index] = self.payoffs[focal_index] + \
                        payoff_focal
                    self.payoffs[other_index] = self.payoffs[other_index] + \
                        payoff_other
        self.payoffs = self.payoffs/samples

    def step(self, samples, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff(samples)

        new_ingroup = []
        new_outgroup = []

        for current_agent in range(self.number_of_agents):
            neighbourhood = [nbr for nbr in
                             self.graph.adj[current_agent].keys()]
            neighbourhood.append(current_agent)

            neighbourhood_payoffs = np.array([self.payoffs[i]
                                              for i in neighbourhood])

            neighbourhood_fitness_sum = np.sum(np.exp(
                selection_intensity*neighbourhood_payoffs))

            neighbourhood_probabilities = np.exp(
                selection_intensity*neighbourhood_payoffs) / \
                neighbourhood_fitness_sum

            agent_to_imitate = np.random.choice(neighbourhood,
                                                p=neighbourhood_probabilities)

            if np.random.rand() <= perturbation_probability:
                current_agent_new_ingroup = np.random.normal(
                    self.ingroup[agent_to_imitate], perturbation_scale)
                if current_agent_new_ingroup < 0:
                    current_agent_new_ingroup = 0.0
                if current_agent_new_ingroup > 1:
                    current_agent_new_ingroup = 1.0

                current_agent_new_outgroup = np.random.normal(
                    self.outgroup[agent_to_imitate], perturbation_scale)
                if current_agent_new_outgroup < 0:
                    current_agent_new_outgroup = 0.0
                if current_agent_new_outgroup > 1:
                    current_agent_new_outgroup = 1.0
            else:
                current_agent_new_ingroup = self.ingroup[agent_to_imitate]
                current_agent_new_outgroup = self.outgroup[agent_to_imitate]

            new_ingroup.append(current_agent_new_ingroup)
            new_outgroup.append(current_agent_new_outgroup)

        self.ingroup = np.array(new_ingroup)
        self.outgroup = np.array(new_outgroup)

    def run_simulation(self, random_seed, number_of_steps, rounds_per_step,
                       selection_intensity, perturbation_probability,
                       perturbation_scale, data_recording,
                       data_file_path, write_frequency):

        np.random.seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)

                for current_step in range(number_of_steps):
                    self.step(rounds_per_step, selection_intensity,
                              perturbation_probability, perturbation_scale)

                    if current_step % write_frequency == 0 \
                            or current_step == number_of_steps - 1:
                        writer.writerow(np.append(self.payoffs,
                                                  np.append(self.ingroup,
                                                            self.outgroup)))
        else:
            for _ in range(number_of_steps):
                self.step(rounds_per_step, selection_intensity,
                          perturbation_probability, perturbation_scale)


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
                  graph)
    model.run_simulation(config["random_seed"], config["number_of_steps"],
                         config["rounds_per_step"],
                         config["selection_intensity"],
                         config["perturbation_probability"],
                         config["perturbation_scale"], config["data_recording"],
                         config["data_file_path"], config["write_frequency"])


if __name__ == "__main__":
    main(sys.argv[1])
