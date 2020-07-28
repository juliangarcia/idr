import numpy as np

def choose_strategy_beliefs(tag, ingroup, outgroup, game, opponent_tag):
    if tag == opponent_tag:
        # Ingroup interaction

        choice_0_value = np.dot(game[0],
                                np.array([ingroup,
                                            1.0 - ingroup]))
        choice_1_value = np.dot(game[1],
                                np.array([ingroup,
                                            1.0 - ingroup]))
    else:
        # Outgroup interaction

        choice_0_value = np.dot(game[0],
                                np.array([outgroup,
                                            1.0 - outgroup]))
        choice_1_value = np.dot(game[1],
                                np.array([outgroup,
                                            1.0 - outgroup]))

    return 0 if choice_0_value > choice_1_value else 1

def choose_strategy_cooperate(tag, ingroup, outgroup, game, opponent_tag):
    return 0

def choose_strategy_defect(tag, ingroup, outgroup, game, opponent_tag):
    return 1

def choose_strategy_direct_action(tag, ingroup, outgroup, game, opponent_tag):

    if tag == opponent_tag:
        # Ingroup interaction

        strategy = np.random.choice([0,1], p=[ingroup, 1-ingroup])

    else:
        # Outgroup interaction

        strategy = np.random.choice([0,1], p=[outgroup, 1-outgroup])

    return strategy

choose_strategy_map = {
    "beliefs": choose_strategy_beliefs,
    "cooperate": choose_strategy_cooperate,
    "defect": choose_strategy_defect,
    "direct_action": choose_strategy_direct_action
}