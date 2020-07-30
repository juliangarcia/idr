#/usr/bin/python


import re
import numpy as np

# UTILS


def find_between(s, pre, post=''):
    result = re.search(pre + '(.*)' + post, s)
    return result.group(1)


def remplazar(diccionario, nombre_template):
    f = open(nombre_template)
    template = f.read()
    for key, value in diccionario.items():
        template = template.replace(str(key), str(value))
    return template

# SPECIFICS

TEMPLATE_NAME = 'TEMPLATE.json'

norms = [9, 0, 12, 8, 10]
learning_rates = [0.01]


# 15 reps
#LR 0.0001, 0.001 
#EXOGENOUS_NORM {0, 9}
# LEARNERS {1...10}


games = {'SH': (4, 1, 3, 2), 'PD': (4, 1, 5, 2)}

beliefs = [(0.99, 0.01), (0.9, 0.9), (0.5, 0.5)]
choose_strategies = ['beliefs', 'direct_action']
graphs = ['gerrymandered', 'scale_free', 'two_communities']

def create_filenames():
    ans = []
    blueprint = 'test_evo_graphs_chooseStrategy_{}_graph_{}_ingroup_{}_outgroup_{}_seed_{}'
    for choose_strategy in choose_strategies:
        for ingroup, outgroup in beliefs:
            for graph in graphs:
                for _ in range(10):
                    seed = np.random.randint(999999999)
                    ans.append(blueprint.format(choose_strategy, graph, ingroup, outgroup, seed))
    return ans

# INFER VARIABLES


def from_file_name_create_dict(filename):
    # ''first_batch_game_{}_seed_{}'
    choose_strategy = find_between(filename, '_chooseStrategy_', '_graph_')
    graph = find_between(filename, '_graph_', '_ingroup_')
    ingroup = find_between(filename, '_ingroup_', '_outgroup_')
    outgroup = find_between(filename, '_outgroup_', '_seed_')
    seed = int(find_between(filename, '_seed_'))
    a = dict()
    a["SEED"] = seed
    a["OUTPUT_FILE_NAME"] = filename + '.csv'
    a["INGROUP"] = ingroup
    a["OUTGROUP"] = outgroup
    a["STRATEGY"] = choose_strategy
    a["GRAPH"] = graph + '.json'
    return a


SUB_FILE_NAME = 'LIST'
TEMPLATE_LINE = 'python model.py {}\n'


def create_sub(list_of_files):
    for filename in list_of_files:
        open(SUB_FILE_NAME, "a").write(
            TEMPLATE_LINE.format(filename + ".json"))


def main():
    list_of_files = create_filenames()
    for filename in list_of_files:
        diccionario = from_file_name_create_dict(filename)
        json_name = filename + '.json'
        json_file = open(json_name, 'w+')
        json_file.write(remplazar(diccionario, TEMPLATE_NAME))
        print('Created ' + json_name)
        json_file.close()
    create_sub(list_of_files)


if __name__ == '__main__':
    main()
