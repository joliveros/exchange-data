import alog
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow_estimator.python.estimator.estimator import Estimator


class ProfitAndLossHook(SessionRunHook):

    estimator: Estimator

    def __init__(self, estimator):
        super(SessionRunHook).__init__()
        self.estimator = estimator
        self.capital = 0

    def begin(self):
        # output = self.estimator.get_variable_value('dense_1/softmax')
        # alog.info(output)
        # alog.info(alog.pformat(self.estimator.get_variable_names()))
        pass

    # def before_run(self, run_context):

    def after_run(self, run_context, run_values):
        # raise Exception()
        # alog.info(run_context)
        # alog.info(run_values)
        # loss_value = run_values.results
        # print("loss value:", loss_value)
        pass
