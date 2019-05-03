import inspect
import sys

from stringcase import pascalcase


class RayAgent(object):
    def __init__(
        self,
        checkpoint_id,
        result_path,
        agent_cls: str,
        **kwargs
    ):
        checkpoint = result_path + \
            f'/checkpoint_{checkpoint_id}' \
            f'/checkpoint-{checkpoint_id}/'

        kwargs['checkpoint'] = checkpoint
        self.checkpoint_id = checkpoint_id


        if isinstance(agent_cls, str):
            classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

            agent_cls = pascalcase(agent_cls)
            agent_cls = [
                cls[1] for cls in classes
                if agent_cls in cls[0]
            ][0]
        self.agent = agent_cls(**kwargs)
