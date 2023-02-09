import unittest

from ..reinforce import ReinforceAgent

class TestReinforceAgent(unittest.TestCase):

    def setUp(self):
        self.reinforce_agent = ReinforceAgent(4, 2)

    def test_reinforce_agent(self):
        self.assertEqual(self.reinforce_agent.state_size, 4)
        self.assertEqual(self.reinforce_agent.action_size, 2)
    
    def test_reinforce_agent_sample_action(self):
        self.assertEqual(self.reinforce_agent.sample_action([1, 2, 3, 4]), 1)

    def test_reinforce_agent_update(self):
        self.assertEqual(self.reinforce_agent.update(), None)
    
    def test_reinforce_agent_save_model(self):
        self.assertEqual(self.reinforce_agent.save_model(), None)

    def test_reinforce_agent_load_model(self):
        self.assertEqual(self.reinforce_agent.load_model(), None)

    def test_reinforce_agent_plot_reward_per_episode(self):
        self.assertEqual(self.reinforce_agent.plot_reward_per_episode(), None)

if __name__ == '__main__':
    unittest.main()