#!/usr/bin/env python
from collections import deque
from os.path import realpath
from pathlib import Path

from transformers import ViTFeatureExtractor, ViTForImageClassification
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

from exchange_data.data import OrderBookFrame
from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame

from PIL import Image as im
from datasets import Dataset
import alog
import click
import numpy as np
import pandas as pd

from exchange_data.data.orderbook_dataset import orderbook_dataset
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

PATH = realpath("./vit_output/pretrained")

model = ViTForImageClassification.from_pretrained(PATH)


class PredictionEmitter(Messenger, BinanceUtils):
    def __init__(self, tick, **kwargs):
        self._kwargs = kwargs
        group_by = kwargs["group_by"]

        alog.info(group_by)
        # if kwargs["futures"]:
        #     exchange = "binance.com-futures"
        # else:
        #     exchange = "binance.com"

        super().__init__(**kwargs)
        BinanceUtils.__init__(self, **kwargs)

        if tick:
            self.tick()
        else:
            self.on(group_by, self.tick)

    def tick(self):
        ds = orderbook_dataset(split=False, **self._kwargs)
        image = ds[-1]["pixel_values"]

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        alog.info(logits)

        predicted_class_idx = logits.argmax(-1).item()

        alog.info(predicted_class_idx)


@click.command()
@click.option("--futures", "-F", is_flag=True)
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
    PredictionEmitter(**kwargs)


if __name__ == "__main__":
    main()
