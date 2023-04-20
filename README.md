# exchange-data

```shell![Order Book Levels](https://user-images.githubusercontent.com/5973528/233488784-78a93f32-2cb3-4340-8f81-ce11f4b3d78c.png)

Saves bitmex data in realtime to influxdb

# Dataframe

1. Set `DB` environment variable with your credentials for InfluxDB instance:


`DB=http://USER:PASSWORD@influxdb:28736/`
```

1. Get a dataframe with `best_bid`, `best_ask` and `orderbook_img` which is an array of the orderbook levels.

```python
from exchange_data.data.orderbook_frame import OrderBookFrame

df = OrderBookFrame().frame

print(df)
```
Output
```shell

best_ask  best_bid                                      orderbook_img                    dtype  macd_diff
time
2023-04-20 18:53:30+00:00  5.003906  5.003906  [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.5], [0...  <class 'numpy.float16'>        NaN
2023-04-20 18:54:00+00:00  5.015625  5.011719  [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0...  <class 'numpy.float16'>        NaN
2023-04-20 18:54:30+00:00  5.011719  5.011719  [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0...  <class 'numpy.float16'>        NaN
2023-04-20 18:55:00+00:00  5.007812  5.003906  [[[0.0], [0.5], [0.5], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>        NaN
2023-04-20 18:55:30+00:00  5.003906  5.003906  [[[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>        NaN
...                             ...       ...                                                ...                      ...        ...
2023-04-20 20:51:30+00:00  5.039062  5.035156  [[[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>        NaN
2023-04-20 20:52:00+00:00  5.031250  5.031250  [[[0.5], [0.5], [0.0], [0.0], [0.0], [0.5], [0...  <class 'numpy.float16'>  -0.001544
2023-04-20 20:52:30+00:00  5.031250  5.027344  [[[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>        NaN
2023-04-20 20:53:00+00:00  5.027344  5.027344  [[[0.5], [0.0], [0.0], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>  -0.002028
2023-04-20 20:53:30+00:00  5.027344  5.023438  [[[0.5], [0.0], [0.5], [0.5], [0.5], [0.5], [0...  <class 'numpy.float16'>        NaN

[241 rows x 5 columns]
```

# Numpy Array

1. Set `DB` environment variable with your credentials for InfluxDB instance:

```shell
`DB=http://USER:PASSWORD@influxdb:28736/`
```

1. Numpy Array

```python
from exchange_data.data.orderbook_frame import OrderBookFrame

obook = df.orderbook_img.to_numpy()

print(obook)
```
Output
```shell

[[0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

```

# Deployment with Docker Stack

1. Deploy Metrics

```shell
docker stack deploy -c docker-compose.yml metrics
```

1. Deploy Databases
```shell
docker stack deploy -c record-docker-compose.yml record
```

1. Deploy
```shell
 docker stack deploy -c docker-compose-binance-futures.yml binance
```

2. Confirm Services are running:

```shell

ID             NAME                          MODE         REPLICAS   IMAGE                                               PORTS
am9ah3oy5qkp   binance_orderbook             replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
308lomi3ita0   binance_orderbook_writer      replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
hx07krnpxw2e   binance_orderbook_writer_1m   replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
kpsmxmqlqjy4   binance_symbol                replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
lqy68b8s34jt   binance_ticker                replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
mkr4969mktkq   binance_trade                 replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
aw61ifnela4k   binance_trade_writer          replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
k0upe3gi36mb   metrics_grafana               replicated   1/1        grafana/grafana-oss:latest                          *:3001->3000/tcp
2tp089ms7r3i   metrics_telegraf              replicated   1/1        telegraf:1.17.2-alpine                              *:8125->8125/udp
kf3fru6n0xeq   record_emit_time              replicated   1/1        registry.rubercubic.com:5001/exchange-data:latest
2smnvztjthpb   record_influxdb               replicated   1/1        registry.rubercubic.com:5001/influxdb:latest        *:28736->8086/tcp
rpiusza8h624   record_redis                  replicated   1/1        redis:latest                                        *:6379->6379/tcp

```
