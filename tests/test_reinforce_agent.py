import pytest

from ..reinforce.reinforce_agent import ReinforceAgent

def test_reinforce_agent():
    reinforce_agent = ReinforceAgent()
    assert reinforce_agent is not None
    assert reinforce_agent.policy_network is not None

def test_reinforce_agent_sample_action():
    reinforce_agent = ReinforceAgent()
    assert reinforce_agent is not None
    assert reinforce_agent.sample_action(1) is not None
    assert reinforce_agent.sample_action(1) <= 1
    assert reinforce_agent.sample_action(1) >= 0
    assert reinforce_agent.sample_action(1) != 0.5

def test_reinforce_agent_update():
    reinforce_agent = ReinforceAgent()
    assert reinforce_agent is not None
    assert reinforce_agent.update() is None

def test_renforce_agent_save_model():
    reinforce_agent = ReinforceAgent()
    assert reinforce_agent is not None
    assert reinforce_agent.save_model() is None