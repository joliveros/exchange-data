import alog
from tgym.envs.orderbook.trade.short import ShortTrade


class TestTradeExecutor(object):

    def test(self):
        trade = ShortTrade(
            entry_price=1.0,
            trading_fee=4e-4,
            min_change=1e-3,
            short_reward_enabled=True
        )

        for i in range(0, 3):
            trade.step(1.0, 1.001)
            alog.info(trade.reward)

        trade.step(1.001, 1.002)
        alog.info(trade.reward)

        trade.step(1.0, 1.001)
        alog.info(trade.reward)

        trade.close()
        alog.info(trade.reward)

        alog.info(alog.pformat(trade.pnl_history.tolist()))

        # assert trade.pnl_history.tolist() == \
        #        [-0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0004]

