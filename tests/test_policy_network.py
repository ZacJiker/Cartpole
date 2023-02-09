import unittest

from ..reinforce import PolicyNetwork

class TestPolicyNetwork(unittest.TestCase):

    def setUp(self):
        self.policy_network = PolicyNetwork(4, 2)

    def test_policy_network(self):
        self.assertEqual(self.policy_network.state_size, 4)
        self.assertEqual(self.policy_network.action_size, 2)

    def test_policy_network_forward(self):
        self.assertEqual(self.policy_network.forward([1, 2, 3, 4]), 2)

if __name__ == '__main__':
    unittest.main()