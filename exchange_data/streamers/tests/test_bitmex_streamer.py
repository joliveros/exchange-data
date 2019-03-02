from dateutil import parser
from exchange_data.streamers._bitmex import BitmexStreamer

import pytest


class TestBitmexStreamer(object):

    @pytest.mark.vcr('once')
    def test_compose_window(self):
        start_date = parser.parse('2019-02-25T03:36:15.751397')

        streamer = BitmexStreamer(
            window_size='1m',
            start_date=start_date,
            influxdb='http://jose:jade121415@0.0.0.0:28953/'
        )

        window = streamer.compose_window()

        assert window.shape == (3, 60, 2, 2, 10)

