import numpy as np
from numba import jit
import csv


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
                 tag1_initial_ingroup_belief, tag1_initial_outgroup_belief):

        # 0 is cooperate
        # 1 is defect

        # Game payoff matrix
        self.game = np.array([[R, S], [T, P]])


        # let us assume pops are even
        assert number_of_agents % 2 == 0

        # could there be a loners option? non-interaction seems relevant
        self.number_of_agents = number_of_agents

        # these contain probabilities that the in-out group
        # will play strategy 0 - cooperate

        self.tags = np.ones(number_of_agents, dtype=int)
        self.ingroup = np.full(number_of_agents, tag1_initial_ingroup_belief, dtype=float)
        self.outgroup = np.full(number_of_agents, tag1_initial_outgroup_belief, dtype=float)

        for i in range(self.number_of_agents // 2):
            self.tags[i] = 0
            self.ingroup[i] = tag0_initial_ingroup_belief
            self.outgroup[i] = tag0_initial_outgroup_belief

        self.matching_indices = list(range(self.number_of_agents))
        self.payoffs = np.zeros(number_of_agents, dtype=float)

        # Data Recording
        self.avg_payoff_0_time_series = []
        self.avg_payoff_1_time_series = []
        self.avg_ingroup_0_time_series = []
        self.avg_ingroup_1_time_series = []
        self.avg_outgroup_0_time_series = []
        self.avg_outgroup_1_time_series = []

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

    def record_data(self):
        # Record the average payoff for each group
        self.avg_payoff_0_time_series.append(
            np.sum(self.payoffs[0:self.number_of_agents // 2]) / len(self.payoffs[0:self.number_of_agents // 2]))
        self.avg_payoff_1_time_series.append(
            np.sum(self.payoffs[self.number_of_agents // 2:] / len(self.payoffs[self.number_of_agents // 2:])))

        # Record average in/outgroup beliefs for each group
        self.avg_ingroup_0_time_series.append(
            np.sum(self.ingroup[0:self.number_of_agents // 2]) / len(self.ingroup[0:self.number_of_agents // 2]))
        self.avg_ingroup_1_time_series.append(
            np.sum(self.ingroup[self.number_of_agents // 2:] / len(self.ingroup[self.number_of_agents // 2:])))

        self.avg_outgroup_0_time_series.append(
            np.sum(self.outgroup[0:self.number_of_agents // 2]) / len(self.outgroup[0:self.number_of_agents // 2]))
        self.avg_outgroup_1_time_series.append(
            np.sum(self.outgroup[self.number_of_agents // 2:] / len(self.outgroup[self.number_of_agents // 2:])))

    def step(self, samples, selection_intensity, perturbation_probability=0.05, perturbation_scale=0.05,
             performance_independent=True):
        # Compute the current payoff
        self.compute_payoff(samples)

        self.record_data()

        # Find the fitness probability distribution (using exponential selection intensity) for each group
        payoff_sum_0 = np.sum(np.exp(selection_intensity*self.payoffs[0:self.number_of_agents // 2]))
        payoff_sum_1 = np.sum(np.exp(selection_intensity*self.payoffs[self.number_of_agents // 2:]))

        fitness_probabilities_0 = np.exp(selection_intensity*self.payoffs[0:self.number_of_agents//2])/payoff_sum_0
        fitness_probabilities_1 = np.exp(selection_intensity*self.payoffs[self.number_of_agents//2:])/payoff_sum_1

        new_ingroup = []
        new_outgroup = []
        new_payoffs = []

        # Create a new generation of agents.  Sampling occurs within group only, to maintain group balance.
        for i in range(self.number_of_agents):
            if i < self.number_of_agents//2:
                current_agent = np.random.choice(range(self.number_of_agents // 2), p=fitness_probabilities_0)
            else:
                current_agent = np.random.choice(range(self.number_of_agents // 2, self.number_of_agents),
                                                 p=fitness_probabilities_1)

            # If performance independent belief perturbation is required, do so
            if performance_independent:
                if np.random.rand() <= perturbation_probability:
                    delta_in = np.random.normal(0.0, perturbation_scale)
                    while self.ingroup[current_agent] + delta_in < 0 or self.ingroup[current_agent] + delta_in > 1:
                        delta_in = np.random.normal(0.0, perturbation_scale)

                    delta_out = np.random.normal(0.0, perturbation_scale)
                    while self.outgroup[current_agent] + delta_out < 0 or self.outgroup[current_agent] + delta_out > 1:
                        delta_out = np.random.normal(0.0, perturbation_scale)
                else:
                    delta_in = 0.0
                    delta_out = 0.0

                new_ingroup.append(self.ingroup[current_agent] + delta_in)
                new_outgroup.append(self.outgroup[current_agent] + delta_out)
            # Else just add the new agent, perturbations is completed by below
            else:
                new_ingroup.append(self.ingroup[current_agent])
                new_outgroup.append(self.outgroup[current_agent])

            new_payoffs.append(self.payoffs[current_agent])

        self.ingroup = np.array(new_ingroup)
        self.outgroup = np.array(new_outgroup)
        self.payoffs = np.array(new_payoffs)

        # If performance dependent perturbation is required, do so.
        if not performance_independent:
            # Compute fitness array
            total_payoffs_sum = np.sum(np.exp(selection_intensity*self.payoffs))
            self.fitness = np.exp(selection_intensity*self.payoffs)/total_payoffs_sum

            # Work out the 25% and 75% percentiles of fitnesses
            p_25 = np.percentile(self.fitness, 25)
            p_75 = np.percentile(self.fitness, 75)

            # Update agents beliefs/strategies
            for i in range(self.number_of_agents):

                if self.fitness[i] < p_25:
                # Agent is in the bottom 25% so update their ingroup and outgroup strategies by sampling 
                # from a Gaussian distribution with mean of the agent's current theta and standard deviation of 5%
                    current_ingroup_strategy = self.ingroup[i]
                    s = -1
                    while 0 > s or s > 1:
                        s = np.random.normal(current_ingroup_strategy, 0.05, 1)
                    self.ingroup[i] = s
                    current_outgroup_strategy = self.outgroup[i]
                    s = -1
                    while 0 > s or s > 1:
                        s = np.random.normal(current_outgroup_strategy, 0.05, 1)
                    self.outgroup[i] = s

                if p_25 <= self.fitness[i] <= p_75:
                # Agent is in the middle 50% so update their ingroup and outgroup strategies by sampling 
                # from a Gaussian distribution with mean of the agent's current theta and standard deviation of 2.5%
                    current_ingroup_strategy = self.ingroup[i]
                    s = -1
                    while 0 > s or s > 1:
                        s = np.random.normal(current_ingroup_strategy, 0.025, 1)
                    self.ingroup[i] = s
                    current_outgroup_strategy = self.outgroup[i]
                    s = -1
                    while 0 > s or s > 1:
                        s = np.random.normal(current_outgroup_strategy, 0.025, 1)
                    self.outgroup[i] = s

                # Agent is in the top 25% so do not update their strategies/beliefs

    def run_simulation(self, number_of_steps, rounds_per_step, selection_intensity, perturbation_probability, perturbation_scale, data_recording=False, performance_independent=True, data_file_name='data.csv'):
        if data_recording:
            with open('data/' + data_file_name, 'w', newline='\n') as out_file:
                writer = csv.writer(out_file)
                for _ in range(number_of_steps):
                    self.step(rounds_per_step, selection_intensity, perturbation_probability, perturbation_scale, performance_independent)
                    writer.writerow(self.payoffs)
        else:
            for _ in range(number_of_steps):
                self.step(rounds_per_step, selection_intensity, perturbation_probability, perturbation_scale, performance_independent)


def main(number_of_agents, R, S, T, P, tag0_in, tag0_out, tag1_in, tag1_out, number_of_steps, rounds_per_step, selection_intensity, perturbation_probability, perturbation_scale, data_recording=False):
    model = Model(number_of_agents, R, S, T, P, tag0_in, tag0_out, tag1_in, tag1_out)
    model.run_simulation(number_of_steps, rounds_per_step, selection_intensity, perturbation_probability, perturbation_scale, data_recording, False)

