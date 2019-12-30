import alog
from matplotlib import pyplot as plt

from exchange_data.utils import Base


class PlotOrderbook(Base):
    def __init__(self, frame_width, top_limit=1e4, **kwargs):
        self.frame_width = frame_width

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(1, 1),
                               dpi=self.frame_width)
        self.top_limit = top_limit

        self.fig = fig
        self.ax1 = ax

        super().__init__(**kwargs)

    def hide_ticks_and_values(self, frame):
        frame.axis('off')
