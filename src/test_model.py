import pytest

import numpy as np
import model


def test_constant_zero_dist():
    m = model.Model()

    # 10 steps, both accelerate constantly of 1
    commands = np.ones((10, 2))

    for ag, env in commands:
        m.step([ag], [env], 0.1)

    dist = m.traces['dist']

    assert np.all(np.isclose(dist, 0))


def test_save_restore_config():
    m = model.Model()

    # 10 steps, both accelerate constantly of 1
    commands = np.ones((10, 2))

    for ag, env in commands:
        m.step([ag], [env], 0.1)

    c = m.save()

    for ag, env in commands:
        m.step([ag], [env], 0.1)

    new_c = m.save()
    assert str(c) != str(new_c)
    m.restore(c)
    old_c = m.save()
    assert str(c) == str(old_c)

    assert m.agent.position != 0
    c['agent']['position'] = 10
    m.restore(c)
    assert m.agent.position == 10


def test_robustness_computation():
    m = model.Model()    
    rc = model.RobustnessComputer('G[1,5](dist > 1 & dist < 10)')

    # 10 steps, agent stays on place, leader moves
    commands = np.ones((10, 2)) * (1, 0)

    for env, ag in commands:
        m.step([env], [ag], 0.1)

    rho = rc.compute(m)
    assert rho < 0

    rho = rc.compute(m, 5)
    assert rho > 0
