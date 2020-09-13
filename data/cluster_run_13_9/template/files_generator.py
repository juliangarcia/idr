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

TEMPLATE_EVO = 'TEMPLATE_EVO.json'
TEMPLATE_EVO_NETWORK = 'TEMPLATE_EVO_NETWORK.json'
TEMPLATE_RL = 'TEMPLATE_RL.json'
TEMPLATE_RL_NETWORK = 'TEMPLATE_RL_NETWORK.json'
'''
beliefs = [0.9, 0.5, 0.1]
payoff_seps = {'small': (4, 1, 3, 2), 'large': (6, 1, 3, 2)}
choose_strategies = ['beliefs', 'direct_action']
graph_types = ['scale_free', 'two_communities']
network_model_types = ['evo_network', 'RL_network']
non_network_model_types = ['evo', 'RL']
'''
beliefs = [0.9,0.1]
payoff_seps = {'small': (4, 1, 3, 2)}
choose_strategies = ['beliefs', 'direct_action']
graph_types = ['two_communities']
network_model_types = ['evo_network', 'RL_network']
non_network_model_types = ['evo', 'RL']

def create_filenames():
    ans = []
    ans_network = []

    # NETWORK files
    blueprint = 'model_{}_graph_{}_chooseStrategy_{}_payoffSep_{}_ingroup_{}_outgroup_{}_seed_{}'
    for model_type in network_model_types:
        for graph_type in graph_types:
            for choose_strategy in choose_strategies:
                for payoff_sep in payoff_seps:
                    for ingroup in beliefs:
                        for outgroup in beliefs:
                            for _ in range(20):
                                seed = np.random.randint(999999999)
                                ans_network.append(blueprint.format(model_type, graph_type, choose_strategy, payoff_sep, ingroup, outgroup, seed))

    # NON-NETWORK files
    blueprint = 'model_{}_chooseStrategy_{}_payoffSep_{}_ingroup_{}_outgroup_{}_seed_{}'
    for model_type in non_network_model_types:
        for choose_strategy in choose_strategies:
            for payoff_sep in payoff_seps:
                for ingroup in beliefs:
                    for outgroup in beliefs:
                        for _ in range(20):
                            seed = np.random.randint(999999999)
                            ans.append(blueprint.format(model_type, choose_strategy, payoff_sep, ingroup, outgroup, seed))

    return ans, ans_network

# INFER VARIABLES


def from_file_name_create_dict(filename, network):
    a = dict()
    if network == True:
        model_type = find_between(filename, 'model_', '_graph_')
        graph_type = find_between(filename, '_graph_', '_chooseStrategy_')
        a["GRAPH_TYPE"] = graph_type
    else:
        model_type = find_between(filename, 'model_', '_chooseStrategy_')
    choose_strategy = find_between(filename, '_chooseStrategy_', '_payoffSep_')
    payoff_sep = find_between(filename, '_payoffSep_', '_ingroup_')
    ingroup = find_between(filename, '_ingroup_', '_outgroup_')
    outgroup = find_between(filename, '_outgroup_', '_seed_')
    seed = int(find_between(filename, '_seed_'))
    r_value, s_value, t_value, p_value = payoff_seps[payoff_sep]
    a["SEED"] = seed
    a["OUTPUT_FILE_NAME"] = filename + '.csv'
    a["MODEL_TYPE"] = model_type
    a["INGROUP"] = ingroup
    a["OUTGROUP"] = outgroup
    a["R_VALUE"] = r_value
    a["S_VALUE"] = s_value
    a["T_VALUE"] = t_value
    a["P_VALUE"] = p_value
    a["STRATEGY"] = choose_strategy

    return a


SUB_FILE_NAME = 'LIST'
TEMPLATE_LINE = 'python model.py {}\n'


def create_sub(list_of_files):
    for filename in list_of_files:
        open(SUB_FILE_NAME, "a").write(
            TEMPLATE_LINE.format(filename + ".json"))


def main():
    list_of_files, list_of_network_files = create_filenames()
    print("done", len(list_of_files), len(list_of_network_files))
    for filename in list_of_files:
        diccionario = from_file_name_create_dict(filename, False)
        json_name = filename + '.json'
        json_file = open(json_name, 'w+')
        if diccionario['MODEL_TYPE'] == 'evo':
            json_file.write(remplazar(diccionario, TEMPLATE_EVO))
        if diccionario['MODEL_TYPE'] == 'RL':
            json_file.write(remplazar(diccionario, TEMPLATE_RL))
        print('Created ' + json_name)
        json_file.close()
    for filename in list_of_network_files:
        diccionario = from_file_name_create_dict(filename, True)
        json_name = filename + '.json'
        json_file = open(json_name, 'w+')
        if diccionario['MODEL_TYPE'] == 'evo_network':
            json_file.write(remplazar(diccionario, TEMPLATE_EVO_NETWORK))
        if diccionario['MODEL_TYPE'] == 'RL_network':
            json_file.write(remplazar(diccionario, TEMPLATE_RL_NETWORK))
        print('Created ' + json_name)
        json_file.close()
    total_list_of_files = list_of_files + list_of_network_files
    create_sub(total_list_of_files)


if __name__ == '__main__':
    main()
