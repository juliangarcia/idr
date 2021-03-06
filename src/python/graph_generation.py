import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
from numba import jit

@jit(nopython=True)
def permute(matching, n):
    for i in range(n-1):
        j = i + np.random.randint(n-i)
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
        agent_0_colour = "#ff0000" # Red
        agent_1_colour = "#0000ff" # Blue
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

        while (tags[edge1[1]] != tags[edge2[1]] or # make sure target group is the same
                edge1[1] == edge2[1] or # make sure targets are not the same
                edge1[0] == edge2[0] or # make sure origins are not the same
                edge1[0] == edge2[1] or # make sure target of edge2 is not origin of edge 1
                edge2[0] == edge1[1] or # make sure target of edge1 is not origin of edge 2
                (edge1[0], edge2[1]) in edges or # make sure swapped edges do not already exist
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
                [1, 2, 4,5]]
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

    F = nx.disjoint_union(G,H)

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
    "scale_free": lambda number_of_agents, number_of_0_tags: randomly_relabel(scale_free_graph(number_of_agents, number_of_0_tags, 0.41, 0.54, 0.05)),
    "two_communities": lambda number_of_agents, number_of_0_tags: two_communities_graph(number_of_agents, number_of_0_tags, number_of_agents//2, number_of_agents//2)
    "gerrymandered": lambda number_of_agents, number_of_0_tags: gerrymandered_graph(number_of_agents, number_of_0_tags, [[3, 3, 3, 3],[3, 3, 3, 3]], number_of_agents//2)
}