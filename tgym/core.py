

class Env(object):
    """Abstract class for an environment. Simplified OpenAI API.
    """

    def __init__(self):
        self.n_actions = None
        self.state_shape = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (numpy.array): action array

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (str): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self):
        """Render the environment.
        """
        raise NotImplementedError()


class DataGenerator(object):
    """Parent class for a data generator. Do not use directly.
    Overwrite the _generator method to create a custom data generator.
    """

    def __init__(self):
        """Initialisation function. The API (gen_kwargs) should be defined in
        the function _generator.
        """
        self._trainable = False
        # We pass self explicitly since we sometimes override rewind (see csv generator)
        self.rewind()
        self.n_products = len(self.next()) / 2
        self.rewind()

    def _generator(self):
        """Generator function. The keywords arguments entirely defines the API
        of the class. This must have a yield statement.
        """
        raise NotImplementedError()

    def next(self):
        """Return the next element in the generator.

        Args:
            numpy.array: next row of the generator
        """
        try:
            return next(self.generator)
        except StopIteration as e:
            self._iterator_end()
            raise e

    def rewind(self):
        """Rewind the generator.
        """
        self.generator = self._generator()

    def _iterator_end(self):
        """End of iterator logic.
        """
        pass
