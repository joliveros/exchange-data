from exchange_data.agents._checkpoint import AgentCheckPoint

import mock
import numpy as np


class TestAgentCheckPoint(object):

    @mock.patch('ray.rllib.agents.dqn.ApexAgent.__init__')
    @mock.patch('exchange_data.agents._checkpoint.AgentCheckPoint'
                '.set_checkpoint_file')
    @mock.patch('exchange_data.agents._checkpoint.AgentCheckPoint'
                '.compute_action')
    def test_agent_init(self, agent_init_mock, mock_set_checkpoint_file,
                        compute_action_mock):
        agent = AgentCheckPoint()

        agent.compute_action(np.array([]))
