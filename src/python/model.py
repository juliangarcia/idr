import numpy as np
from numba import jit
import csv
import json
import sys
from choose_strategy_functions import *


@jit(nopython=True)
def permute(matching, n):
    for i in range(n-1):
        j = i + np.random.randint(n-i)
        temp = matching[i]
        matching[i] = matching[j]
        matching[j] = temp

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
                 initial_number_of_0_tags, choose_strategy):

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
        # self.payoffs = self.payoffs/samples

    def step(self, samples, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff(samples)

        # Find the fitness probability distribution (using exponential selection intensity) for each group

        payoff_0 = np.array([agent.payoff for agent in self.agents[0:self.number_of_0_tags]])
        payoff_1 = np.array([agent.payoff for agent in self.agents[self.number_of_0_tags:]])

        payoff_sum_0 = np.sum(np.exp(selection_intensity*payoff_0))
        payoff_sum_1 = np.sum(np.exp(selection_intensity*payoff_1))

        fitness_probabilities_0 = np.exp(selection_intensity*payoff_0)/payoff_sum_0
        fitness_probabilities_1 = np.exp(selection_intensity*payoff_1)/payoff_sum_1

        # new_ingroup = []
        # new_outgroup = []
        # new_payoffs = []

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

    def run_simulation(self, random_seed, number_of_steps, rounds_per_step,
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
                    self.step(rounds_per_step, selection_intensity,
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
                self.step(rounds_per_step, selection_intensity,
                          perturbation_probability, perturbation_scale)


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
                  choose_strategy_map[config["choose_strategy"]])
    model.run_simulation(config["random_seed"], config["number_of_steps"],
                         config["rounds_per_step"],
                         config["selection_intensity"],
                         config["perturbation_probability"],
                         config["perturbation_scale"], config["data_recording"],
                         config["data_file_path"], config["write_frequency"])


if __name__ == "__main__":
    main(sys.argv[1])
