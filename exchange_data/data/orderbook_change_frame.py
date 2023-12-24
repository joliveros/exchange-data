#!/usr/bin/env python

from exchange_data.data import OrderBookFrame
import time
import alog
import click
import numpy as np
import pandas as pd


class OrderBookChangeFrame(OrderBookFrame):
    def __init__(self, additional_group_by, show=False, **kwargs):
        super().__init__(**kwargs)
        self.additional_group_by = additional_group_by

        self.show = show

    @property
    def frame(self):
        if self.cache and self.filename.exists():
            df = pd.read_pickle(str(self.filename))
            self.quantile = df.attrs["quantile"]
            self.trade_volume_max = df.attrs["trade_volume_max"]
            self.change_max = df.attrs["change_max"]
            return df

        df = self._frame()

        orderbook_img = df.orderbook_img.to_numpy().tolist()

        df.drop(["orderbook_img"], axis=1)

        orderbook_img = np.asarray(orderbook_img)

        orderbook_img = np.concatenate(
            (orderbook_img[:, :, 0], orderbook_img[:, :, 1]), axis=2
        )

        orderbook_img = np.absolute(orderbook_img)

        for frame_ix in range(0, orderbook_img.shape[0]):
            frame = orderbook_img[frame_ix]
            last_col = None

            for col_ix in range(0, frame.shape[0]):
                col = frame[col_ix]
                col_copy = col.copy()
                col = np.zeros(col.shape)

                if type(last_col) is np.ndarray:
                    for last_price_ix in range(0, last_col.shape[0]):
                        last_price = last_col[last_price_ix]
                        ix = np.where(col_copy[:, :-1] == last_price[0])[0]
                        current_price = col_copy[ix]
                        if current_price.shape[0] > 0:
                            current_price = current_price[0].copy()
                            change = last_price[-1] - current_price[-1]
                            current_price[-1] = change
                            col[ix] = current_price

                    frame[col_ix] = col

                last_col = col_copy

        for frame_ix in range(orderbook_img.shape[0]):
            orderbook = orderbook_img[frame_ix]
            shape = orderbook.shape
            new_ob = np.zeros((shape[0], shape[1], 1))

            last_frame_price = orderbook[-1][:, 0]

            for i in range(shape[0]):
                frame = orderbook[i]

                for l in range(frame.shape[0]):
                    level = frame[l]

                    price, volume = level

                    last_frame_index = np.where(last_frame_price == price)

                    new_ob[i, last_frame_index[0], 0] = np.asarray([volume])

            orderbook_img[frame_ix] = new_ob

        orderbook_img = np.delete(orderbook_img, 1, axis=3)

        df["orderbook_img"] = [
            self.plot_orderbook(np.rot90(np.fliplr(orderbook_img[i])))
            for i in range(0, orderbook_img.shape[0])
        ]

        df = df.resample(self.additional_group_by).last()

        if self.show:
            for ix in range(0, df.shape[0]):
                ob_img = df["orderbook_img"].iloc[ix]
                time.sleep(1 / 3)
                self.show_img(ob_img)

        df.attrs["trade_volume_max"] = self.trade_volume_max
        df.attrs["change_max"] = self.change_max
        df.attrs["quantile"] = self.quantile

        self.cache_frame(df)

        return df


@click.command()
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--depth", default=72, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--offset-interval", "-o", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--tick", is_flag=True)
@click.option("--cache", is_flag=True)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    df = OrderBookChangeFrame(**kwargs).frame

    # alog.info(df)

    # pd.set_option('display.max_rows', len(df) + 1)

    obook = df.orderbook_img.to_numpy()

    obook = np.squeeze(obook[-1])

    alog.info(obook.tolist())


if __name__ == "__main__":
    main()
