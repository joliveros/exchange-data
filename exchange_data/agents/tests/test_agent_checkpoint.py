from exchange_data.agents._apex_agent_check_point import ApexAgentCheckPoint

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
        agent = ApexAgentCheckPoint()

        agent.compute_action(np.array([]))
