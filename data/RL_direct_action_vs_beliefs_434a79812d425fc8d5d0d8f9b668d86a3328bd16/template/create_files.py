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

beliefs = [0.9, 0.5, 0.1]
payoff_seps = {'small': (4, 1, 3, 2), 'large': (6, 1, 3, 2)}
choose_strategies = ['beliefs', 'direct_action']

def create_filenames():
    ans = []
    blueprint = 'RL_direct_action_vs_beliefs_chooseStrategy_{}_payoffSep_{}_ingroup_{}_outgroup_{}_seed_{}'
    for choose_strategy in choose_strategies:
        for payoff_sep in payoff_seps:
            for ingroup in beliefs:
                for outgroup in beliefs:
                    for _ in range(100):
                        seed = np.random.randint(999999999)
                        ans.append(blueprint.format(choose_strategy, payoff_sep, ingroup, outgroup, seed))
    return ans

# INFER VARIABLES


def from_file_name_create_dict(filename):
    # ''first_batch_game_{}_seed_{}'
    choose_strategy = find_between(filename, '_chooseStrategy_', '_payoffSep_')
    payoff_sep = find_between(filename, '_payoffSep_', '_ingroup_')
    ingroup = find_between(filename, '_ingroup_', '_outgroup_')
    outgroup = find_between(filename, '_outgroup_', '_seed_')
    seed = int(find_between(filename, '_seed_'))
    r_value, s_value, t_value, p_value = payoff_seps[payoff_sep]
    a = dict()
    a["SEED"] = seed
    a["OUTPUT_FILE_NAME"] = filename + '.csv'
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
