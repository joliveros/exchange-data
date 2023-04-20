
# Order Book Tradying Gym

`OrderBookFrameEnv` inherits from `OrderBookFrame` which enables it to load the necessary data from the time series database.

## Example Usage:

```python
from tgym.envs import OrderBookFrameEnv
from pytimeparse.timeparse import timeparse
import random

env = OrderBookFrameEnv(
    random_frame_start=False,
    short_reward_enabled=True,
    is_training=False,
)

env.reset()

for i in range(timeparse('1h')):
    env.step(random.randint(0, 1))
```

# OpenAI Gym

```python
import tgym.envs
import gym

env = gym.make('orderbook-frame-env-v0')
```
