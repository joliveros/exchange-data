import alog
import pytest

from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._long_position_emitter import \
    LongPositionEmitter
from exchange_data.utils import DateTimeUtils


class TestLongPositionEmitter(object):

    @pytest.mark.vcr()
    def test_init(self):
        start_date = DateTimeUtils.parse_datetime_str(
            '2019-04-23 02:37:37.887295+00:00')
        emitter = LongPositionEmitter(
            job_name='test',
            symbol=BitmexChannels.XBTUSD,
            start_date=start_date,
            end_date=start_date
        )

        assert emitter.frames[-1].shape == (84, 84)
