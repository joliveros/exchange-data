#!/usr/bin/env python

from datetime import timedelta
from exchange_data.streamers._bitmex import BitmexStreamer
from exchange_data.utils import DateTimeUtils

import alog
import click


class OrderBookLevelStreamer(BitmexStreamer):
    def __init__(self, symbol, depth=40, group_by="2s", **kwargs):
        super().__init__(group_by=group_by, **kwargs)
        self.symbol = symbol
        self.depth = depth
        self.group_by = group_by
        self.last_timestamp = self.start_date
        self.channel_name = self.gen_channel_name()
        self.current_query = self.get_orderbook_frames()

    def gen_channel_name(self):
        if self.depth > 0:
            return f"{self.symbol}_OrderBookFrame_depth_{self.depth}"
        else:
            return f"{self.symbol}_OrderBookFrame"

    def orderbook_frame_query(self):
        start_date = self.start_date
        end_date = self.end_date

        start_date = self.format_date_query(start_date)
        end_date = self.format_date_query(end_date)

        query = (
            f"SELECT last(*) AS data FROM {self.channel_name} "
            f"WHERE time >= {start_date} AND time <= {end_date} GROUP BY "
            f"time({self.group_by});"
        )

        return self.query(query)

    def get_orderbook_frames(self):
        orderbook = self.orderbook_frame_query()

        for data in orderbook.get_points(self.channel_name):
            timestamp = DateTimeUtils.parse_db_timestamp(data["time"])

            if self.last_timestamp != timestamp:
                best_bid = data["data_best_bid"]
                best_ask = data["data_best_ask"]

                levels = data["data_data"]
                self.last_timestamp = timestamp

                if best_ask and best_bid and levels:
                    yield timestamp, best_ask, best_bid, levels

            self.last_timestamp = timestamp

    def send(self, *args):
        if self.last_timestamp < self.stop_date:
            try:
                return next(self.current_query)
            except StopIteration as e:
                self._set_next_window()

                if self.end_date > self.stop_date:
                    raise StopIteration()
                self.current_query = self.get_orderbook_frames()
                return None, None, None, None

        raise StopIteration()


@click.command()
@click.option(
    "--window-size", "-w", type=str, default="1m", help='Window size i.e. "1m"'
)
@click.option(
    "--sample-interval",
    "-s",
    type=str,
    default="1s",
    help="interval at which to sample data from db.",
)
def main(**kwargs):
    end_date = DateTimeUtils.now()
    start_date = end_date - timedelta(seconds=60)

    streamer = OrderBookLevelStreamer(
        database_name="bitmex", end_date=end_date, start_date=start_date, **kwargs
    )

    for timestamp, best_ask, best_bid, orderbook_img in streamer:
        alog.info((str(timestamp), best_ask, best_bid))


if __name__ == "__main__":
    main()
