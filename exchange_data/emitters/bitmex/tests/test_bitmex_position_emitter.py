import mock
import pytest


from exchange_data.emitters.bitmex._bitmex_position_emitter import \
    BitmexPositionEmitter
from exchange_data.utils import DateTimeUtils


class TestBitmexPositionEmitter(DateTimeUtils):

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_initialize_position_emitter(self, signal_mock):
        start_date = self.parse_datetime_str(
            '2019-03-17 18:17:56.571810+00:00')
        end_date = self.parse_datetime_str(
            '2019-03-17 20:44:36.571810+00:00')
        emitter = BitmexPositionEmitter(
            job_name='test',
            start_date=start_date,
            end_date=end_date,
            agent_cls=mock.MagicMock
        )

        assert emitter.last_observation.shape[0] == \
               emitter.observation_space.shape[0]
