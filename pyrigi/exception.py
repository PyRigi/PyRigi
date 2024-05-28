class LoopError(ValueError):
    def __init__(self, msg="The graph needs to be loop-free.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
