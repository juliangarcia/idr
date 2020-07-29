import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json



def gerrymandered_graph(rewire_amount):
    number_of_agents = 24
    initial_number_of_0_tags = number_of_agents//2

    tags = np.ones(number_of_agents, dtype=int)

    for i in range(initial_number_of_0_tags):
        tags[i] = 0

    districts = [[3, 3, 3, 3],
                 [3, 3, 3, 3]]

    district_assignment = [0 for _ in range(number_of_agents)]

    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(number_of_agents)])

    for current_agent in range(number_of_agents):
        if current_agent < initial_number_of_0_tags:
            current_agent_tag = 0
        else:
            current_agent_tag = 1

        district = np.random.randint(0, 4)
        while districts[current_agent_tag][district] == 0:
            district = np.random.randint(0, 4)

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

        while tags[edge1[1]] != tags[edge2[1]] or \
                edge1[1] == edge2[1] or \
                edge1[0] == edge2[0] or \
                edge1[0] == edge2[1] or \
                edge2[0] == edge1[1] or \
                (edge1[0], edge2[1]) in edges or \
                (edge2[0], edge1[1]) in edges:
            edge1 = edges[np.random.choice(range(len(edges)))]
            edge2 = edges[np.random.choice(range(len(edges)))]

        G.remove_edge(*edge1)
        G.remove_edge(*edge2)
        G.add_edge(edge1[0], edge2[1])
        G.add_edge(edge2[0], edge1[1])

    nx.draw(G, with_labels=True)
    plt.show()

    with open("graph.json", 'w') as out_file:
        json.dump(nx.readwrite.json_graph.node_link_data(G), out_file, indent=4)


#gerrymandered_graph(100)

def Two_communities_graph(number_of_agents, initial_number_of_0_tags, rewire_amount):

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


    nx.draw(F, with_labels=True)
    plt.show()

    with open("graph.json", 'w') as out_file:
        json.dump(nx.readwrite.json_graph.node_link_data(F), out_file, indent=4)


#Two_communities_graph(40, 10, 4)
