


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


ingroup_beliefs = [0.9, 0.5, 0.1]
outgroup_beliefs = [0.9, 0.5, 0.1]


def create_filenames():
    ans = []
    blueprint = 'test_personal_strategy_ingroup_{}_outgroup_{}_seed_{}'
    for ingroup in ingroup_beliefs:
        for outgroup in outgroup_beliefs:
            for _ in range(10):
                seed = np.random.randint(999999999)
                ans.append(blueprint.format(ingroup, outgroup, seed))
    return ans

# INFER VARIABLES


def from_file_name_create_dict(filename):
    # ''first_batch_game_{}_seed_{}'
    ingroup = find_between(filename, '_ingroup_', '_outgroup_')
    outgroup = find_between(filename, '_outgroup_', '_seed_')
    seed = int(find_between(filename, '_seed_'))
    a = dict()
    a["SEED"] = seed
    a["OUTPUT_FILE_NAME"] = filename + '.csv'
    a["INGROUP"] = ingroup
    a["OUTGROUP"] = outgroup
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
