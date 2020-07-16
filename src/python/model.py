import numpy as np
from numba import jit
import csv
import json


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

    def step(self, samples, selection_intensity, perturbation_probability=0.05,
             perturbation_scale=0.05):

        # Compute the current payoff
        self.compute_payoff(samples)

        self.record_data()

        # Find the fitness probability distribution (using exponential selection intensity) for each group
        payoff_sum_0 = np.sum(np.exp(selection_intensity*self.payoffs[0:self.number_of_0_tags]))
        payoff_sum_1 = np.sum(np.exp(selection_intensity*self.payoffs[self.number_of_0_tags:]))

        fitness_probabilities_0 = np.exp(selection_intensity*self.payoffs[0:self.number_of_0_tags])/payoff_sum_0
        fitness_probabilities_1 = np.exp(selection_intensity*self.payoffs[self.number_of_0_tags:])/payoff_sum_1

        new_ingroup = []
        new_outgroup = []
        new_payoffs = []

        # Create a new generation of agents.  Sampling occurs within
        # group only, to maintain group balance.
        for i in range(self.number_of_agents):
            if i < self.number_of_0_tags:
                current_agent = np.random.choice(range(self.number_of_0_tags),
                                                 p=fitness_probabilities_0)
            else:
                current_agent = np.random.choice(range(self.number_of_0_tags, self.number_of_agents),
                                                 p=fitness_probabilities_1)

            # Perturb agents belief
            if np.random.rand() <= perturbation_probability:
                current_agent_new_ingroup = np.random.normal(
                    self.ingroup[current_agent], perturbation_scale)
                if current_agent_new_ingroup < 0:
                    current_agent_new_ingroup = 0.0
                if current_agent_new_ingroup > 1:
                    current_agent_new_ingroup = 1.0

                current_agent_new_outgroup = np.random.normal(
                    self.outgroup[current_agent], perturbation_scale)
                if current_agent_new_outgroup < 0:
                    current_agent_new_outgroup = 0.0
                if current_agent_new_outgroup > 1:
                    current_agent_new_outgroup = 1.0
            else:
                current_agent_new_ingroup = self.ingroup[current_agent]
                current_agent_new_outgroup = self.outgroup[current_agent]

            new_ingroup.append(current_agent_new_ingroup)
            new_outgroup.append(current_agent_new_outgroup)
            new_payoffs.append(self.payoffs[current_agent])

        self.ingroup = np.array(new_ingroup)
        self.outgroup = np.array(new_outgroup)
        self.payoffs = np.array(new_payoffs)

    def run_simulation(self, random_seed, number_of_steps, rounds_per_step,
                       selection_intensity, perturbation_probability,
                       perturbation_scale, data_recording,
                       data_file_path_payoffs, data_file_path_ingroup, data_file_path_outgroup):

        random.seed(random_seed)

        if data_recording:
            with open(data_file_path_payoffs, 'w', newline='\n') as f, 
                 open(data_file_path_ingroup, 'w', newline='\n') as g,
                 open(data_file_path_outgroup, 'w', newline='\n') as h:
                writer1 = csv.writer(f)
                writer2 = csv.writer(g)
                writer3 = csv.writer(h)
                for _ in range(number_of_steps):
                    self.step(rounds_per_step, selection_intensity,
                              perturbation_probability, perturbation_scale)
                    writer1.writerow(self.payoffs)
                    writer2.writerow(self.ingroup)
                    writer3.writerow(self.outgroup)

        else:
            for _ in range(number_of_steps):
                self.step(rounds_per_step, selection_intensity,
                          perturbation_probability, perturbation_scale)


def main(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    model = Model(**config["model_initialisation"])
    model.run_simulation(**config["simulation_settings"])
