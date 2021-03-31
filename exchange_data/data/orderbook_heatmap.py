#!/usr/bin/env python
from dash.dependencies import Output, Input

from exchange_data.data.orderbook_frame import OrderBookFrame

import alog
import click
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'


class OrderBookHeatMap(OrderBookFrame):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs)
        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(id='chart', style={'height':'100vh'}),
            dcc.Interval(
                id='interval-component',
                interval=10*1000, # in milliseconds
                n_intervals=0
            )
        ])

        @app.callback(Output('chart', 'figure'),
                      Input('interval-component', 'n_intervals'))
        def update_graph_live(n):
            df = self.price_level_frame

            orderbook = df.orderbook_img.to_list()
            orderbook = np.concatenate(orderbook)
            orderbook = np.squeeze(orderbook)
            shape = orderbook.shape
            new_ob = np.zeros((shape[0], shape[1]))

            last_frame = orderbook[-1]
            last_frame = last_frame[:, 0]
            last_frame = np.sort(last_frame)

            for i in range(shape[0]):
                frame = orderbook[i]
                for l in range(frame.shape[0]):
                    price, volume = frame[l]
                    last_frame_index = np.where(last_frame == price)
                    new_ob[i, last_frame_index[0]] = volume

            orderbook = np.fliplr(new_ob)
            orderbook = np.rot90(orderbook)

            alog.info(orderbook)

            fig = go.Figure(data=go.Heatmap(
                    z=orderbook,
                    x=df.index.to_pydatetime(),
                    y=[],
                    colorscale='Deep'))

            alog.info(fig)

            return fig

        app.run_server(debug=False, use_reloader=False)

@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--tick', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    OrderBookHeatMap(**kwargs)


if __name__ == '__main__':
    main()
