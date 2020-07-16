import pytest
import numpy as np
from model import Model


def test_initialisation():
    model = Model(10, -1, -3, 0, -2, 1, 0, 1, 0, 4)

    # test game matrix
    assert model.game.shape == (2, 2)
    assert type(model.game[0][0]) is np.int64
    assert type(model.game[0][1]) is np.int64
    assert type(model.game[1][0]) is np.int64
    assert type(model.game[1][1]) is np.int64

    # test number of 0 tags is less than the total number of agents
    assert model.number_of_0_tags < model.number_of_agents

    # test the lengths of the arrays are correct
    assert len(model.tags) == model.number_of_agents
    assert len(model.ingroup) == model.number_of_agents
    assert len(model.outgroup) == model.number_of_agents
    assert len(model.matching_indices) == model.number_of_agents
    assert len(model.payoffs) == model.number_of_agents

    # test that in/outgroup beliefs are all floats and in the range [0,1]
    for i in range(model.number_of_agents):
        assert type(model.ingroup[i]) is np.float64
        assert type(model.outgroup[i]) is np.float64
        assert 0 <= model.ingroup[i] <= 1
        assert 0 <= model.outgroup[i] <= 1


def test_step():
    # Test that initial group balance is maintained.
    # Changes initial numbers of tag 0 agents
    for i in range(1, 10):
        model = Model(10, -1, -3, 0, -2, 1, 0, 1, 0, i)
        for _ in range(10):
            model.step(10, 0.1)
            assert np.count_nonzero(model.tags == 0) == i
            assert np.count_nonzero(model.tags == 1) == 10-i

    # Test that agents beliefs are constrained in [0,1]
    model = Model(10, -1, -3, 0, -2, 1, 0, 1, 0, 5)
    for _ in range(10):
        model.step(10, 0.1)
        assert np.amax(model.ingroup) <= 1.0
        assert np.amin(model.ingroup) >= 0.0

        assert np.amax(model.outgroup) <= 1.0
        assert np.amin(model.outgroup) >= 0.0
