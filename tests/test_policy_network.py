import pytest

from ..reinforce.policy_network import PolicyNetwork

def test_policy_network():
    policy_network = PolicyNetwork()
    assert policy_network is not None
    assert policy_network.shared_net is not None


def test_policy_network_forward():
    policy_network = PolicyNetwork()
    assert policy_network is not None
    assert policy_network.forward(1) is not None
    assert policy_network.forward(1).shape == (1, 2)
    assert policy_network.forward(1).sum() == 1
    assert policy_network.forward(1).max() <= 1