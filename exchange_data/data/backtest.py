#!/usr/bin/env python
from exchange_data.data import OrderBookFrame
from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame
from exchange_data.emitters.binance import BinanceUtils
from os.path import realpath
from pathlib import Path
from transformers import (
    ViTFeatureExtractor,
    ViTForImageClassification,
    ViTImageProcessor,
)
from PIL import Image as im

import datasets
import alog
import click
import numpy as np


class Backtest(BinanceUtils):
    def __init__(self, **kwargs):
        alog.info(alog.pformat(kwargs))

        self._kwargs = kwargs
        group_by = kwargs["group_by"]
        super().__init__(**kwargs)

        BinanceUtils.__init__(self, **kwargs)

        ob_frame = OrderBookChangeFrame(**kwargs)
        df = ob_frame.frame

        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        device = "cuda:0"

        PATH = realpath("../vit_output/pretrained")
        alog.info(PATH)
        model = ViTForImageClassification.from_pretrained(PATH)
        model = model.to(device)

        df["prediction"] = None
        predictions = []
        for ix in range(0, df.shape[0]):
            image = df.iloc[ix]["orderbook_img"]
            image = im.fromarray(image)

            inputs = feature_extractor(images=image, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()

            predictions.append(predicted_class_idx)

        df["prediction"] = np.asarray(predictions)

        self.frame = df


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
@click.option("--cache", is_flag=True)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    Backtest(**kwargs)


if __name__ == "__main__":
    main()
