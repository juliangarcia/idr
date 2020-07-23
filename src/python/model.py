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


class Agent:
    def __init__(self, tag, ingroup, outgroup):
        self.tag = tag
        self.ingroup = ingroup
        self.outgroup = outgroup
        self.payoff = 0.0


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

        self.agents = [
            Agent(1, tag1_initial_ingroup_belief, tag1_initial_outgroup_belief)
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

        if agent_focal.tag == agent_other.tag:
            # ingroup interaction

            # choice focal
            choice_0_value = np.dot(self.game[0],
                                    np.array([agent_focal.ingroup,
                                              1.0 - agent_focal.ingroup]))
            choice_1_value = np.dot(self.game[1],
                                    np.array([agent_focal.ingroup,
                                              1.0 - agent_focal.ingroup]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            # choice other
            choice_0_value = np.dot(self.game[0],
                                    np.array([agent_other.ingroup,
                                              1.0 - agent_other.ingroup]))
            choice_1_value = np.dot(self.game[1],
                                    np.array([agent_other.ingroup,
                                              1.0 - agent_other.ingroup]))
            choice_other = 0 if choice_0_value > choice_1_value else 1

            return self.game[choice_focal, choice_other], \
                   self.game[choice_other, choice_focal]

        else:
            # outgroup interaction

            # choice focal
            choice_0_value = np.dot(self.game[0],
                                    np.array([agent_focal.outgroup,
                                              1.0 - agent_focal.outgroup]))
            choice_1_value = np.dot(self.game[1],
                                    np.array([agent_focal.outgroup,
                                              1.0 - agent_focal.outgroup]))
            choice_focal = 0 if choice_0_value > choice_1_value else 1

            # choice other
            choice_0_value = np.dot(self.game[0],
                                    np.array([agent_other.outgroup,
                                             1.0 - agent_other.outgroup]))
            choice_1_value = np.dot(self.game[1],
                                    np.array([agent_other.outgroup,
                                              1.0 - agent_other.outgroup]))
            choice_other = 0 if choice_0_value > choice_1_value else 1

            return self.game[choice_focal, choice_other], \
                   self.game[choice_other, choice_focal]

    def compute_payoff(self):
        # self.payoffs = np.zeros(self.number_of_agents)
        for agent in self.agents:
            agent.payoff = 0.0

        for focal_agent_index in range(self.number_of_agents):
            if len(self.graph.adj[focal_agent_index].keys()) > 0:

                neighbours = [nbr for nbr in self.graph.adj[focal_agent_index].keys()]

                for neighbour_index in neighbours:
                    payoff_focal, _ = self.encounter(self.agents[focal_agent_index], self.agents[neighbour_index])
                    self.agents[focal_agent_index].payoff += payoff_focal
                self.agents[focal_agent_index].payoff /= len(neighbours)

    def step(self, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff()

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
                      current_agent_new_ingroup, current_agent_new_outgroup))

        self.agents = new_agents

    def run_simulation(self, random_seed, number_of_steps,
                       selection_intensity, perturbation_probability,
                       perturbation_scale, data_recording,
                       data_file_path, write_frequency):

        np.random.seed(random_seed)

        if data_recording:
            with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)
                header_list = ["P" + str(i) for i in range(self.number_of_agents)] + \
                              ["I" + str(i) for i in range(self.number_of_agents)] + \
                              ["O" + str(i) for i in range(self.number_of_agents)]
                writer.writerow(header_list)

                for current_step in range(number_of_steps):
                    self.step(selection_intensity,
                              perturbation_probability, perturbation_scale)

                    if current_step % write_frequency == 0 \
                            or current_step == number_of_steps - 1:
                        payoffs = np.array([agent.payoff
                                            for agent in self.agents])
                        ingroup = np.array([agent.ingroup
                                            for agent in self.agents])
                        outgroup = np.array([agent.outgroup
                                             for agent in self.agents])
                        writer.writerow(np.append(payoffs,
                                                  np.append(ingroup,
                                                            outgroup)))
        else:
            for _ in range(number_of_steps):
                self.step(selection_intensity,
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
                         config["selection_intensity"],
                         config["perturbation_probability"],
                         config["perturbation_scale"], config["data_recording"],
                         config["data_file_path"], config["write_frequency"])


if __name__ == "__main__":
    main(sys.argv[1])
