from datetime import timedelta, datetime
from dateutil import parser, tz
from exchange_data.streamers._bitmex import BitmexStreamer

import alog
import mock
import pytest


class TestBitmexStreamer(object):
    start_date_str = '2019-03-02 20:04:48.468323'

    start_date = parser.parse(start_date_str).replace(tzinfo=tz.tzlocal())

    min_date = parser.parse('2019-02-20 23:15:24.665000-06:00')\
        .replace(tzinfo=tz.tzlocal())

    @pytest.mark.vcr()
    def test_first_date_available(self):
        end_date = self.start_date + timedelta(seconds=2)

        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=end_date
        )


        assert streamer.min_date == self.min_date

        orderbook_frames = streamer.orderbook_frame_query()
        last_timestamp = None

        for frame in orderbook_frames.get_points(streamer.channel_name):
            last_timestamp = \
                datetime.utcfromtimestamp(frame['time'] / 1000)\
                .replace(tzinfo=tz.tzlocal())

        assert end_date > last_timestamp > self.start_date

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.BitmexStreamer.now',
        return_value=start_date
    )
    def test_init_with_window_size(self, now_mock):
        streamer = BitmexStreamer(
            window_size='2s'
        )

        assert streamer.min_date == self.min_date

        orderbook_frames = streamer.orderbook_frame_query()

        last_timestamp = None
        frame_count = 0

        for frame in orderbook_frames.get_points(streamer.channel_name):
            last_timestamp = \
                datetime.utcfromtimestamp(frame['time'] / 1000)\
                .replace(tzinfo=tz.tzlocal())

            alog.debug(last_timestamp)

            frame_count += 1

        window_size = streamer.end_date - streamer.start_date

        assert window_size.seconds == 2
        assert frame_count == 2
        assert streamer.start_date < last_timestamp < streamer.end_date

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.BitmexStreamer.now',
        return_value=start_date
    )
    def test_init_with_window_size(self, now_mock):
        streamer = BitmexStreamer(
            window_size='2s'
        )

        assert streamer.min_date == self.min_date

        orderbook_frames = streamer.orderbook_frame_query()

        last_timestamp = None
        frame_count = 0

        for frame in orderbook_frames.get_points(streamer.channel_name):
            last_timestamp = \
                datetime.utcfromtimestamp(frame['time'] / 1000)\
                .replace(tzinfo=tz.tzlocal())

            alog.debug(last_timestamp)

            frame_count += 1

        window_size = streamer.end_date - streamer.start_date

        assert window_size.seconds == 2
        assert frame_count == 2
        assert streamer.start_date < last_timestamp < streamer.end_date

    @pytest.mark.vcr()
    def test_compose_window(self):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(minutes=1),
        )

        index, orderbook = streamer.compose_window()

        assert index.shape == (60,)
        assert orderbook.shape == (60, 2, 2, 10)

    @pytest.mark.vcr()
    def test_one_second_window(self):
        streamer = BitmexStreamer(
            start_date=self.start_date,
            end_date=self.start_date + timedelta(seconds=1),
        )

        index, orderbook = streamer.compose_window()

        assert index.shape == (1,)
        assert orderbook.shape == (1, 2, 2, 10)

    @pytest.mark.vcr()
    def test_random_start_date_for_window(self):
        streamer = BitmexStreamer(
            random_start_date=True,
            end_date=self.start_date,
            window_size='1s'
        )

        assert streamer.start_date != \
               parser.parse('2019-02-24 19:07:10.543897-06:00')

