import alog
from matplotlib import pyplot as plt


class PlotOrderbook(object):
    def __init__(self, frame_width, **kwargs):
        self.frame_width = frame_width

        plt.close()

        fig, frames = plt.subplots(2, 1, figsize=(1, 1),
                                   dpi=self.frame_width)

        ax1, ax2 = frames
        self.fig = fig
        self.ax1 = ax2
        self.ax2 = ax1
        # self.ax2 = fig.add_subplot(1, 2, 2, frame_on=False)

    def hide_ticks_and_values(self, frame):
        frame.axis('off')
