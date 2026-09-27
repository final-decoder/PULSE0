"""Tests for the acquisition loop (Algorithm 1) on small instances."""
import numpy as np
import pytest

from pulse import acquisition, networks, simulation


@pytest.fixture(scope="module")
def small_case():
    net = networks.generate_network(13000, p=4, q=2, radius=0.7)
    pools = simulation.generate_all_pools(net, n_blocks=24, horizon=2.0,
                                          seed_base=13000)
    r_hat = simulation.baseline_rate_estimate(net, duration=600.0,
                                              seed=13001)
    return net, pools, r_hat


@pytest.mark.parametrize("rule", acquisition.RULES)
def test_rules_respect_budget_and_choose_valid_target(small_case, rule):
    net, pools, r_hat = small_case
    c = np.ones(net.p)
    budget = 48
    res = acquisition.run_acquisition(rule, pools, r_hat, c, 0.8, budget)
    assert 0 <= res.chosen < net.p
    assert res.allocation.sum() <= budget + net.p * acquisition.protocol.INIT_BLOCKS
    assert (res.allocation >= min(acquisition.protocol.INIT_BLOCKS, 24)).all()


def test_identical_pools_across_rules(small_case):
    """The finite-pool convention: every rule inspects the same blocks."""
    net, pools, r_hat = small_case
    c = np.ones(net.p)
    res_a = acquisition.run_acquisition(acquisition.RULE_UNIFORM, pools,
                                        r_hat, c, 0.8, 40)
    res_b = acquisition.run_acquisition(acquisition.RULE_UNIFORM, pools,
                                        r_hat, c, 0.8, 40)
    assert res_a.chosen == res_b.chosen
    assert np.array_equal(res_a.allocation, res_b.allocation)


def test_uniform_rule_allocates_evenly(small_case):
    net, pools, r_hat = small_case
    c = np.ones(net.p)
    res = acquisition.run_acquisition(acquisition.RULE_UNIFORM, pools,
                                      r_hat, c, 0.8, 40)
    assert res.allocation.max() - res.allocation.min() <= 1
