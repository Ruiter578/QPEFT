class EngineBase:
    def __init__(self, logger, args):
        self.logger = logger
        self.args = args

    def __call__(self, *args, **kwds):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError